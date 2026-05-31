from tools.import_manipulation_dataset import import_dataset, load_dataset_config, plan_dataset_import


def test_ycb_dataset_import_dry_run_lists_objects(tmp_path) -> None:
    summary = import_dataset(
        config_path="configs/datasets/ycb_core_subset.yaml",
        output_dir=tmp_path / "ycb",
        object_ids=["003_cracker_box"],
        dry_run=True,
    )

    assert summary["dataset_id"] == "ycb_core_subset_v1"
    assert summary["dry_run"] is True
    assert summary["objects"][0]["object_id"] == "003_cracker_box"
    assert summary["objects"][0]["downloaded"] is False


def test_dataset_plan_filters_object_ids() -> None:
    config = load_dataset_config("configs/datasets/ycb_core_subset.yaml")
    plan = plan_dataset_import(config, object_ids=["006_mustard_bottle"])

    assert [item["object_id"] for item in plan] == ["006_mustard_bottle"]
    assert plan[0]["archive_name"].endswith(".tgz")
