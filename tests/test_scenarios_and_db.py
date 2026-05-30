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

