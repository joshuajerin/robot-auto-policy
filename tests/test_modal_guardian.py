from tools.modal_guardian import load_phase1_specs, scan_for_modal_errors


def test_scan_for_modal_errors_matches_failure_lines() -> None:
    text = """
    [info] training iteration 15/1000
    RuntimeError: CUDA out of memory while allocating tensor
    Traceback (most recent call last):
    """

    hits = scan_for_modal_errors(text)

    assert any("CUDA out of memory" in hit for hit in hits)
    assert any("Traceback" in hit for hit in hits)


def test_scan_for_modal_errors_ignores_headless_isaac_warnings() -> None:
    text = """
    [Warning] [omni.platforminfo.plugin] failed to open the default display.
    Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid'
    [INFO] training continues
    """

    assert scan_for_modal_errors(text) == []


def test_load_phase1_specs_accepts_single_spec(tmp_path) -> None:
    path = tmp_path / "spec.json"
    path.write_text('{"experiment_id":"quick"}')

    assert load_phase1_specs(path) == [{"experiment_id": "quick"}]


def test_load_phase1_specs_accepts_batch(tmp_path) -> None:
    path = tmp_path / "specs.json"
    path.write_text('[{"experiment_id":"a"},{"experiment_id":"b"}]')

    assert [spec["experiment_id"] for spec in load_phase1_specs(path)] == ["a", "b"]
