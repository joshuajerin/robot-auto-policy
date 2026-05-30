from modal_runner.modal_app import _fallback_eval_metrics, _score_metrics


def test_fallback_eval_metrics_fail_safely() -> None:
    raw = _fallback_eval_metrics("phase1_quick", "eval did not write metrics")
    score = _score_metrics(raw)

    assert raw["evaluation_errors"] == ["eval did not write metrics"]
    assert score["safety_passed"] is False
    assert "evaluation emitted errors" in score["safety_reasons"]
    assert score["total_score"] == 0.0
