import json
from pathlib import Path

from modal_runner.deployed import load_specs


def test_load_specs_accepts_single_object(tmp_path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"experiment_id": "one"}))

    assert load_specs(path) == [{"experiment_id": "one"}]


def test_load_specs_accepts_list(tmp_path) -> None:
    path = tmp_path / "specs.json"
    path.write_text(json.dumps([{"experiment_id": "one"}, {"experiment_id": "two"}]))

    assert [spec["experiment_id"] for spec in load_specs(path)] == ["one", "two"]
