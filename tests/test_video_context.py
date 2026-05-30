from pathlib import Path

from agents.video_context import summarize_locomotion_video


def test_video_context_requires_existing_file(tmp_path) -> None:
    missing = tmp_path / "missing.mp4"
    try:
        summarize_locomotion_video(missing)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_video_context_summary_for_existing_file(tmp_path) -> None:
    fake_video = tmp_path / "walk.webm"
    fake_video.write_bytes(b"not a real video but enough for existence fallback")

    summary = summarize_locomotion_video(fake_video, source_url="https://example.com/walk.webm")

    assert summary["style"] == "upright human walk"
    assert summary["target_velocity_class"] == "normal_walk"
    assert summary["source_url"] == "https://example.com/walk.webm"
