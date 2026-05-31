import json

from agents.video_motion_imitation import build_h1_retarget_map, prepare_user_video_motion_skill


def test_h1_retarget_map_covers_major_limbs() -> None:
    robot_spec = json.loads(open("assets/h1_robot_spec.json").read())

    retarget_map = build_h1_retarget_map(robot_spec)

    assert retarget_map["robot_id"] == "unitree_h1"
    assert "left_knee" in retarget_map["mapping"]
    assert retarget_map["mapping"]["left_knee"]["robot_joints"] == ["left_knee"]
    assert "right_shoulder" in retarget_map["mapping"]


def test_prepare_user_video_motion_skill_writes_context(tmp_path) -> None:
    video = tmp_path / "walk.mp4"
    video.write_bytes(b"not a real mp4 but enough for metadata fallback")

    result = prepare_user_video_motion_skill(video_path=video, output_dir=tmp_path / "skill")

    context_path = tmp_path / "skill" / "motion_imitation_context.json"
    assert result["motion_imitation_context"] == str(context_path)
    context = json.loads(context_path.read_text())
    assert context["source_type"] == "user_recorded_mp4"
    assert context["style_context"]["source_type"] == "user_video_motion_imitation"
    assert context["pose_estimation"]["required"] is True
    assert context["safety"]["llm_direct_torque_control"] is False
