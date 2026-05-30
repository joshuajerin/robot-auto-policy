from adapters.base import ExperimentHistory
from adapters.locomotion.scenario_generator import classify_frontier, generate_locomotion_scenarios
from core.experiment_db import ExperimentDB
from core.schemas import ScenarioSpec


def test_frontier_classification() -> None:
    assert classify_frontier(0.95) == "too_easy"
    assert classify_frontier(0.5) == "learning_frontier"
    assert classify_frontier(0.2) == "too_hard"


def test_generator_prioritizes_push_failures() -> None:
    history = ExperimentHistory(failure_reports=[{"failure_report_json": "fails_on_push"}])
    scenarios = generate_locomotion_scenarios(history)
    assert scenarios[0].scenario_id == "side_push_recovery_v001"


def test_generator_mutates_frontier_continuously() -> None:
    history = ExperimentHistory(
        scenario_matrix=[
            {
                "scenario_id": "low_friction_walk_v001",
                "difficulty": 0.25,
                "success_rate": 0.94,
            },
            {
                "scenario_id": "side_push_recovery_v001",
                "difficulty": 0.35,
                "success_rate": 0.52,
            },
            {
                "scenario_id": "rough_heightfield_walk_v001",
                "difficulty": 0.50,
                "success_rate": 0.08,
            },
        ]
    )

    scenarios = generate_locomotion_scenarios(history)
    ids = {scenario.scenario_id for scenario in scenarios}

    assert "low_friction_walk_v002" in ids
    assert "uphill_walk_v001" in ids
    assert "rough_heightfield_walk_v002" in ids
    assert all(scenario.parent_scenario_id for scenario in scenarios[:3])
    assert {scenario.difficulty for scenario in scenarios}


def test_generator_advances_from_leaf_scenarios() -> None:
    history = ExperimentHistory(
        scenario_matrix=[
            {
                "scenario_id": "low_friction_walk_v001",
                "parent_scenario_id": None,
                "difficulty": 0.25,
                "success_rate": 0.52,
            },
            {
                "scenario_id": "side_push_recovery_v001",
                "parent_scenario_id": "low_friction_walk_v001",
                "difficulty": 0.35,
                "success_rate": 0.55,
            },
        ]
    )

    scenarios = generate_locomotion_scenarios(history)

    assert all(scenario.parent_scenario_id != "low_friction_walk_v001" for scenario in scenarios)
    assert any(scenario.parent_scenario_id == "side_push_recovery_v001" for scenario in scenarios)


def test_experiment_db_persists_scenarios(tmp_path) -> None:
    db = ExperimentDB(tmp_path / "research.db")
    db.insert_scenarios(
        [
            ScenarioSpec(
                scenario_id="low_friction_walk_test",
                task_family="locomotion",
                difficulty=0.2,
            )
        ]
    )

    matrix = db.scenario_matrix()
    db.close()
    assert matrix[0]["scenario_id"] == "low_friction_walk_test"
