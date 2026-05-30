"""Render a real Unitree H1 walking video with Blender on Modal.

This is intentionally separate from Isaac Lab rendering. Isaac Sim camera
rendering needs a Vulkan/RTX graphics stack that Modal containers may not expose
reliably. Blender's Python package can render the public Unitree H1 USD bundle
with CUDA/Cycles on Modal GPU workers, so it gives us an actual model video for
review while Isaac Lab remains the source of policy training.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import modal


APP_NAME = "robogenesis-blender-h1-render"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_H1_ASSET_ROOT = Path("/robogenesis/assets/unitree_h1")
DEFAULT_TRACE = "artifacts/downloaded_modal/baseline_h1_cmu_walk_telemetry-seed-200/rollout_trace.json"
DEFAULT_OUTPUT = "artifacts/blender_renders/h1_blender_walk.mp4"


app = modal.App(APP_NAME)

rendering_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "xorg",
        "libegl1",
        "libgl1",
        "libxi6",
        "libxkbcommon0",
        "libxrender1",
        "libxfixes3",
        "libxext6",
        "libx11-6",
        "libsm6",
    )
    .uv_pip_install("bpy==4.5.0")
)
try:
    rendering_image = rendering_image.add_local_dir(REPO_ROOT / "assets" / "unitree_h1", remote_path=str(REMOTE_H1_ASSET_ROOT))
except AttributeError:
    pass


@app.function(
    gpu="RTX-PRO-6000",
    image=rendering_image,
    cpu=8.0,
    memory=32_768,
    timeout=60 * 60,
    max_containers=1,
)
def render_h1_walk_video(
    trace_json: str = "{}",
    *,
    frame_count: int = 72,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
    samples: int = 32,
) -> bytes:
    import bpy

    trace = json.loads(trace_json or "{}")
    _clear_scene(bpy)
    _configure_rendering(bpy, width=width, height=height, samples=samples)

    usd_path = REMOTE_H1_ASSET_ROOT / "usd" / "configuration" / "h1_base.usd"
    if not usd_path.exists():
        raise FileNotFoundError(f"Missing H1 USD asset at {usd_path}")

    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=str(usd_path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError(f"Blender imported no objects from {usd_path}")

    root = bpy.data.objects.new("RoboGenesis_H1_Root", None)
    bpy.context.collection.objects.link(root)
    for obj in imported:
        if obj.parent is None:
            obj.parent = root

    _wire_h1_hierarchy(imported)
    _center_model_on_ground(imported, root)
    _create_world(bpy)
    _animate_walk(bpy, root, imported, trace=trace, frame_count=frame_count, fps=fps)
    _ground_root_animation(bpy, root, imported, frame_count=frame_count)
    _create_camera_and_lights(bpy)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frame_count
    bpy.context.scene.frame_set(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bpy.context.scene.render.filepath = str(tmp / "frame_")
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(animation=True)

        output_path = tmp / "h1_blender_walk.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )
        return output_path.read_bytes()


def _clear_scene(bpy: Any) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _configure_rendering(bpy: Any, *, width: int, height: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    cycles = bpy.context.preferences.addons["cycles"]
    try:
        cycles.preferences.compute_device_type = "OPTIX"
        scene.cycles.device = "GPU"
        cycles.preferences.get_devices()
        for device in cycles.preferences.devices:
            device.use = device.type != "CPU"
            print(f"Blender device: {device.name} {device.type} use={device.use}", flush=True)
    except Exception as exc:
        try:
            cycles.preferences.compute_device_type = "CUDA"
            scene.cycles.device = "GPU"
            cycles.preferences.get_devices()
            for device in cycles.preferences.devices:
                device.use = device.type != "CPU"
                print(f"Blender device: {device.name} {device.type} use={device.use}", flush=True)
        except Exception as cuda_exc:
            scene.cycles.device = "CPU"
            print(
                "[WARN] GPU rendering unavailable, using CPU: "
                f"{type(exc).__name__}: {exc}; CUDA fallback: {type(cuda_exc).__name__}: {cuda_exc}",
                flush=True,
            )


def _center_model_on_ground(imported: list[Any], root: Any) -> None:
    bounds = _world_bounds(imported)
    center_x = (bounds["min_x"] + bounds["max_x"]) * 0.5
    center_y = (bounds["min_y"] + bounds["max_y"]) * 0.5
    min_z = bounds["min_z"]
    root.location.x -= center_x
    root.location.y -= center_y
    root.location.z -= min_z


def _world_bounds(objects: list[Any]) -> dict[str, float]:
    from mathutils import Vector

    values: dict[str, float] = {
        "min_x": float("inf"),
        "max_x": float("-inf"),
        "min_y": float("inf"),
        "max_y": float("-inf"),
        "min_z": float("inf"),
        "max_z": float("-inf"),
    }
    for obj in objects:
        if getattr(obj, "type", None) != "MESH" or getattr(obj, "hide_render", False):
            continue
        if not hasattr(obj, "bound_box"):
            continue
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            values["min_x"] = min(values["min_x"], point.x)
            values["max_x"] = max(values["max_x"], point.x)
            values["min_y"] = min(values["min_y"], point.y)
            values["max_y"] = max(values["max_y"], point.y)
            values["min_z"] = min(values["min_z"], point.z)
            values["max_z"] = max(values["max_z"], point.z)
    if not math.isfinite(values["min_x"]):
        return {"min_x": -0.5, "max_x": 0.5, "min_y": -0.5, "max_y": 0.5, "min_z": 0.0, "max_z": 1.8}
    return values


def _wire_h1_hierarchy(imported: list[Any]) -> None:
    links = _indexed_link_xforms(imported)
    parent_pairs = (
        ("left_hip_roll_link", "left_hip_yaw_link"),
        ("left_hip_pitch_link", "left_hip_roll_link"),
        ("left_knee_link", "left_hip_pitch_link"),
        ("left_ankle_link", "left_knee_link"),
        ("right_hip_roll_link", "right_hip_yaw_link"),
        ("right_hip_pitch_link", "right_hip_roll_link"),
        ("right_knee_link", "right_hip_pitch_link"),
        ("right_ankle_link", "right_knee_link"),
        ("torso_link", "pelvis"),
        ("left_shoulder_pitch_link", "torso_link"),
        ("left_shoulder_roll_link", "left_shoulder_pitch_link"),
        ("left_shoulder_yaw_link", "left_shoulder_roll_link"),
        ("left_elbow_link", "left_shoulder_yaw_link"),
        ("right_shoulder_pitch_link", "torso_link"),
        ("right_shoulder_roll_link", "right_shoulder_pitch_link"),
        ("right_shoulder_yaw_link", "right_shoulder_roll_link"),
        ("right_elbow_link", "right_shoulder_yaw_link"),
    )
    for child_name, parent_name in parent_pairs:
        child = links.get(child_name)
        parent = links.get(parent_name)
        if child is None or parent is None or child.parent is parent:
            continue
        child_world = child.matrix_world.copy()
        child.parent = parent
        child.matrix_world = child_world


def _create_world(bpy: Any) -> None:
    bpy.context.scene.world = bpy.data.worlds.new("RoboGenesis_World")
    bpy.context.scene.world.color = (0.03, 0.035, 0.04)

    floor_mat = bpy.data.materials.new("matte charcoal floor")
    floor_mat.diffuse_color = (0.05, 0.055, 0.06, 1.0)
    bpy.ops.mesh.primitive_plane_add(size=10, location=(1.2, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "ground_plane"
    floor.data.materials.append(floor_mat)

    stripe_mat = bpy.data.materials.new("subtle walking path")
    stripe_mat.diffuse_color = (0.12, 0.16, 0.18, 1.0)
    for x in [-1.0, 0.5, 2.0, 3.5]:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0.0, 0.006))
        stripe = bpy.context.object
        stripe.name = "floor_stride_marker"
        stripe.scale = (0.025, 1.3, 0.004)
        stripe.data.materials.append(stripe_mat)


def _animate_walk(
    bpy: Any,
    root: Any,
    imported: list[Any],
    *,
    trace: dict[str, Any],
    frame_count: int,
    fps: int,
) -> None:
    from mathutils import Euler

    objects = _indexed_link_xforms(imported)
    base_rotations = {obj.name: obj.rotation_euler.copy() for obj in imported}
    trace_frames = trace.get("frames") if isinstance(trace.get("frames"), list) else []
    base_root_z = float(root.location.z)

    root.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    cycles = max(1.25, frame_count / max(1, fps) * 1.75)
    for frame in range(1, frame_count + 1):
        t = (frame - 1) / max(1, frame_count - 1)
        phase = 2.0 * math.pi * cycles * t
        trace_frame = trace_frames[min(len(trace_frames) - 1, int(t * max(0, len(trace_frames) - 1)))] if trace_frames else {}
        trace_joint_pos = trace_frame.get("joint_pos") if isinstance(trace_frame, dict) else []
        if not isinstance(trace_joint_pos, list):
            trace_joint_pos = []

        root.location = (1.2 * t - 0.6, 0.0, base_root_z + 0.02 + 0.035 * abs(math.sin(phase)))
        root.rotation_euler = Euler((0.035 * math.sin(phase + 0.4), 0.02 * math.sin(phase * 0.5), 0.012 * math.sin(phase)), "XYZ")
        root.keyframe_insert(data_path="location", frame=frame)
        root.keyframe_insert(data_path="rotation_euler", frame=frame)

        left = math.sin(phase)
        right = math.sin(phase + math.pi)
        _pose_link(objects, base_rotations, "left_hip_pitch", axis=1, angle=0.24 * left + _trace(trace_joint_pos, 2, 0.06), frame=frame)
        _pose_link(objects, base_rotations, "right_hip_pitch", axis=1, angle=0.24 * right + _trace(trace_joint_pos, 7, 0.06), frame=frame)
        _pose_link(objects, base_rotations, "left_knee", axis=1, angle=0.42 * max(0.0, -left) + _trace(trace_joint_pos, 3, 0.06), frame=frame)
        _pose_link(objects, base_rotations, "right_knee", axis=1, angle=0.42 * max(0.0, -right) + _trace(trace_joint_pos, 8, 0.06), frame=frame)
        _pose_link(objects, base_rotations, "left_ankle", axis=1, angle=-0.16 * left + _trace(trace_joint_pos, 4, 0.04), frame=frame)
        _pose_link(objects, base_rotations, "right_ankle", axis=1, angle=-0.16 * right + _trace(trace_joint_pos, 9, 0.04), frame=frame)
        _pose_link(objects, base_rotations, "left_shoulder_pitch", axis=1, angle=-0.30 * left + _trace(trace_joint_pos, 11, 0.06), frame=frame)
        _pose_link(objects, base_rotations, "right_shoulder_pitch", axis=1, angle=-0.30 * right + _trace(trace_joint_pos, 15, 0.06), frame=frame)
        _pose_link(objects, base_rotations, "left_elbow", axis=1, angle=0.18 + 0.14 * max(0.0, left), frame=frame)
        _pose_link(objects, base_rotations, "right_elbow", axis=1, angle=0.18 + 0.14 * max(0.0, right), frame=frame)


def _ground_root_animation(bpy: Any, root: Any, imported: list[Any], *, frame_count: int) -> None:
    for frame in range(1, frame_count + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        bounds = _world_bounds(imported)
        if not math.isfinite(bounds["min_z"]):
            continue
        root.location.z += 0.015 - bounds["min_z"]
        root.keyframe_insert(data_path="location", frame=frame)


def _indexed_link_xforms(imported: list[Any]) -> dict[str, Any]:
    index: dict[str, Any] = {}
    wanted = (
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    )
    for obj in imported:
        if obj.type == "MESH":
            continue
        normalized = obj.name.split(".", 1)[0]
        if normalized in wanted and normalized not in index:
            index[normalized] = obj
    print(f"Indexed {len(index)} H1 link xforms", flush=True)
    return index


def _pose_link(objects: dict[str, Any], base_rotations: dict[str, Any], key: str, *, axis: int, angle: float, frame: int) -> None:
    obj = objects.get(f"{key}_link")
    if obj is None:
        return
    rotation = base_rotations[obj.name].copy()
    rotation[axis] += angle
    obj.rotation_euler = rotation
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def _trace(values: list[Any], index: int, scale: float) -> float:
    try:
        return float(values[index]) * scale
    except Exception:
        return 0.0


def _create_camera_and_lights(bpy: Any) -> None:
    from mathutils import Vector

    def look_at(obj: Any, target: tuple[float, float, float]) -> None:
        direction = Vector(target) - obj.location
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 4.5))
    key = bpy.context.object
    key.name = "large_softbox"
    key.data.energy = 900
    key.data.size = 4.5

    bpy.ops.object.light_add(type="POINT", location=(-2.2, 1.8, 1.8))
    rim = bpy.context.object
    rim.name = "small_rim_light"
    rim.data.energy = 80

    bpy.ops.object.camera_add(location=(2.5, -4.2, 1.25))
    camera = bpy.context.object
    camera.name = "tracking_camera"
    camera.data.lens = 55
    look_at(camera, (0.05, 0.0, 0.82))
    bpy.context.scene.camera = camera


@app.local_entrypoint()
def main(
    trace_path: str = DEFAULT_TRACE,
    output_path: str = DEFAULT_OUTPUT,
    frame_count: int = 72,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
    samples: int = 32,
) -> None:
    trace_file = Path(trace_path)
    trace_json = trace_file.read_text() if trace_file.exists() else "{}"
    video = render_h1_walk_video.remote(
        trace_json,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        samples=samples,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(video)
    print(f"Blender H1 video saved to {output.resolve()}")
