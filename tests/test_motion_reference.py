from agents.motion_reference import build_motion_context, count_amc_frames, select_motion


def test_count_amc_frames(tmp_path) -> None:
    amc = tmp_path / "walk.amc"
    amc.write_text(
        "\n".join(
            [
                "#!OML:ASF",
                ":FULLY-SPECIFIED",
                "1",
                "root 0 0 0 0 0 0",
                "lowerback 1 2 3",
                "2",
                "root 0 0 0 0 0 0",
            ]
        )
    )

    assert count_amc_frames(amc) == 2


def test_build_motion_context_uses_research_mocap_source(tmp_path) -> None:
    config = {
        "reference_id": "cmu_human_walk_v1",
        "dataset_id": "cmu_graphics_lab_mocap",
        "dataset_name": "CMU Graphics Lab Motion Capture Database",
        "source_homepage": "http://mocap.cs.cmu.edu/",
        "license_note": "free motions",
        "format": "asf_amc",
        "subject_id": "07",
        "style_context": {"style": "research_mocap_normal_walk"},
        "training_use": {"phase1": "style only"},
    }
    motion = {
        "motion_id": "07_01",
        "description": "walk",
        "asf_url": "http://mocap.cs.cmu.edu/subjects/07/07.asf",
        "amc_url": "http://mocap.cs.cmu.edu/subjects/07/07_01.amc",
    }

    context = build_motion_context(
        config=config,
        motion=motion,
        asf_path=tmp_path / "07.asf",
        amc_path=tmp_path / "07_01.amc",
        motion_metadata={"frame_count": 120, "sample_rate_hz": 120.0, "duration_seconds": 1.0},
    )

    assert context["dataset_id"] == "cmu_graphics_lab_mocap"
    assert context["style_context"]["source_type"] == "research_motion_capture"
    assert context["style_context"]["source_motion_id"] == "07_01"


def test_select_motion_reports_unknown_id() -> None:
    config = {"motions": [{"motion_id": "07_01"}]}

    try:
        select_motion(config, "missing")
    except ValueError as exc:
        assert "07_01" in str(exc)
        return
    raise AssertionError("expected ValueError")
