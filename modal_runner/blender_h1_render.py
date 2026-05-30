"""Render a Unitree H1 physics-scene video with Blender on Modal.

This is intentionally separate from Isaac Lab rendering. Isaac Sim camera
rendering needs a Vulkan/RTX graphics stack that Modal containers may not expose
reliably. Blender's Python package can render the public Unitree H1 USD bundle
with CUDA/Cycles on Modal GPU workers, so it gives us an actual model video for
review while Isaac Lab remains the source of policy training.

The renderer imports the top-level ``h1.usd`` asset, whose default variant loads
the PhysX articulation payload. Blender does not execute NVIDIA PhysX
articulation drives, so this path preserves the official connected asset and
scene context for visualization instead of pretending to simulate joint torques.
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
DEFAULT_OUTPUT = "artifacts/blender_renders/h1_physics_scene_walk.mp4"


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

    usd_path = REMOTE_H1_ASSET_ROOT / "usd" / "h1.usd"
    if not usd_path.exists():
        raise FileNotFoundError(f"Missing H1 USD asset at {usd_path}")

    print(f"Importing Unitree H1 physics USD asset: {usd_path}", flush=True)
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

    _tag_asset_metadata(root, usd_path)
    _hide_physics_debug_geometry(imported)
    _center_model_on_ground(imported, root)
    floor = _create_world(bpy)
    _setup_scene_physics(bpy, floor)
    _animate_connected_root_motion(bpy, root, trace=trace, frame_count=frame_count, fps=fps)
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


def _tag_asset_metadata(root: Any, usd_path: Path) -> None:
    root["robogenesis_asset"] = "unitree_h1"
    root["robogenesis_usd_source"] = str(usd_path)
    root["robogenesis_usd_physics_variant"] = "PhysX"
    root["robogenesis_renderer_note"] = "Blender renders the USD PhysX asset; Isaac/Omniverse runs articulated physics."


def _hide_physics_debug_geometry(imported: list[Any]) -> None:
    for obj in imported:
        lineage = " ".join(_object_lineage(obj)).lower()
        if any(token in lineage for token in ("collisions", "collision", "collider")):
            obj.hide_render = True
            obj.hide_viewport = True


def _object_lineage(obj: Any) -> list[str]:
    names: list[str] = []
    current = obj
    while current is not None:
        names.append(str(getattr(current, "name", "")))
        current = getattr(current, "parent", None)
    return names


def _create_world(bpy: Any) -> Any:
    bpy.context.scene.world = bpy.data.worlds.new("RoboGenesis_World")
    bpy.context.scene.world.color = (0.025, 0.027, 0.03)

    floor_mat = bpy.data.materials.new("rubberized lab floor")
    floor_mat.diffuse_color = (0.075, 0.078, 0.076, 1.0)
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.075, 0.078, 0.076, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.86

    bpy.ops.mesh.primitive_cube_add(size=1, location=(1.0, 0.0, -0.015))
    floor = bpy.context.object
    floor.name = "physics_ground_rubber_floor"
    floor.scale = (7.0, 3.0, 0.015)
    floor.data.materials.append(floor_mat)

    lane_mat = bpy.data.materials.new("low profile lane markings")
    lane_mat.diffuse_color = (0.47, 0.55, 0.55, 1.0)
    for y in (-0.42, 0.42):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(1.0, y, 0.006))
        lane = bpy.context.object
        lane.name = "ground_lane_boundary"
        lane.scale = (5.0, 0.01, 0.003)
        lane.data.materials.append(lane_mat)

    for x in [-2.2, -1.0, 0.2, 1.4, 2.6, 3.8]:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0.0, 0.008))
        stripe = bpy.context.object
        stripe.name = "ground_distance_marker"
        stripe.scale = (0.012, 0.42, 0.003)
        stripe.data.materials.append(lane_mat)

    wall_mat = bpy.data.materials.new("matte rear wall")
    wall_mat.diffuse_color = (0.11, 0.12, 0.12, 1.0)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(1.0, 1.53, 1.2))
    wall = bpy.context.object
    wall.name = "lab_back_wall"
    wall.scale = (7.0, 0.025, 1.2)
    wall.data.materials.append(wall_mat)
    return floor


def _setup_scene_physics(bpy: Any, floor: Any) -> None:
    scene = bpy.context.scene
    scene.gravity = (0.0, 0.0, -9.81)
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    scene.rigidbody_world.time_scale = 1.0
    scene.rigidbody_world.substeps_per_frame = 10
    scene.rigidbody_world.solver_iterations = 25

    bpy.ops.object.select_all(action="DESELECT")
    floor.select_set(True)
    bpy.context.view_layer.objects.active = floor
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    floor.rigid_body.friction = 0.95
    floor.rigid_body.restitution = 0.02
    floor.rigid_body.collision_shape = "BOX"
    floor.select_set(False)


def _animate_connected_root_motion(
    bpy: Any,
    root: Any,
    *,
    trace: dict[str, Any],
    frame_count: int,
    fps: int,
) -> None:
    from mathutils import Euler

    trace_frames = trace.get("frames") if isinstance(trace.get("frames"), list) else []
    base_root_z = float(root.location.z)

    root.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    cycles = max(1.0, frame_count / max(1, fps) * 1.25)
    velocity_command = _velocity_command(trace_frames)
    x_travel = max(0.9, min(2.0, velocity_command * frame_count / max(1, fps) * 0.55))
    for frame in range(1, frame_count + 1):
        t = (frame - 1) / max(1, frame_count - 1)
        phase = 2.0 * math.pi * cycles * t
        trace_frame = trace_frames[min(len(trace_frames) - 1, int(t * max(0, len(trace_frames) - 1)))] if trace_frames else {}
        base_ang_vel = trace_frame.get("base_ang_vel") if isinstance(trace_frame, dict) else []
        yaw_rate = _trace(base_ang_vel, 2, 0.05) if isinstance(base_ang_vel, list) else 0.0

        root.location = (x_travel * t - x_travel * 0.5, 0.0, base_root_z + 0.018 + 0.012 * abs(math.sin(phase)))
        root.rotation_euler = Euler((0.018 * math.sin(phase + 0.35), 0.01 * math.sin(phase * 0.5), yaw_rate), "XYZ")
        root.keyframe_insert(data_path="location", frame=frame)
        root.keyframe_insert(data_path="rotation_euler", frame=frame)


def _ground_root_animation(bpy: Any, root: Any, imported: list[Any], *, frame_count: int) -> None:
    for frame in range(1, frame_count + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        bounds = _world_bounds(imported)
        if not math.isfinite(bounds["min_z"]):
            continue
        root.location.z += 0.015 - bounds["min_z"]
        root.keyframe_insert(data_path="location", frame=frame)


def _velocity_command(trace_frames: list[Any]) -> float:
    if not trace_frames:
        return 0.75
    commands: list[float] = []
    for frame in trace_frames[: min(20, len(trace_frames))]:
        if not isinstance(frame, dict):
            continue
        command = frame.get("velocity_command")
        if isinstance(command, list) and command:
            try:
                commands.append(abs(float(command[0])))
            except Exception:
                continue
    if not commands:
        return 0.75
    return sum(commands) / len(commands)


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
