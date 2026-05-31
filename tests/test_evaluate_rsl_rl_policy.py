from modal_runner.isaac_scripts.evaluate_rsl_rl_policy import MetricAccumulator, TraceRecorder


def test_metric_accumulator_reads_direct_env_extras_for_manipulation() -> None:
    accumulator = MetricAccumulator(policy_id="h1_tabletop_eval", seed_count=1)

    accumulator.observe_step(
        obs=None,
        infos={},
        env_extras={
            "log": {
                "task_success_rate": 0.25,
                "task_progress": 0.4,
                "contact_stability": 0.6,
                "placement_accuracy": 0.5,
                "placement_error_m": 0.08,
                "object_slip_rate": 0.1,
                "collision_rate": 0.0,
                "force_violation_rate": 0.0,
                "robot_fall_rate": 0.0,
                "object_drop_rate": 0.0,
            }
        },
        rewards=None,
        actions=None,
        dones=None,
    )
    accumulator.observe_episodes(_FakeTensor([1.0, 2.0]), _FakeTensor([239.0, 240.0]), max_steps=240)

    metrics = accumulator.to_metrics()

    assert metrics["metric_family"] == "manipulation"
    assert metrics["task_success_rate"] == 0.25
    assert metrics["task_progress"] == 0.4
    assert metrics["contact_stability"] == 0.6
    assert metrics["placement_accuracy"] == 0.5
    assert metrics["survival_no_fall"] == 1.0


def test_trace_recorder_labels_h1_tabletop_observations() -> None:
    recorder = TraceRecorder(policy_id="h1_tabletop_eval", env_index=0, max_steps=1)
    obs = _FakeTensor([[float(index) for index in range(79)]])
    actions = _FakeTensor([[0.1] * 19])
    rewards = _FakeTensor([1.5])
    dones = _FakeTensor([0.0])

    recorder.observe(
        seed=101,
        step=1,
        obs=obs,
        rewards=rewards,
        actions=actions,
        dones=dones,
        infos={},
        env_extras={"log": {"task_progress": 0.3}},
    )

    assert recorder.frames[0]["trace_family"] == "h1_tabletop_manipulation"
    assert recorder.frames[0]["cube_relative_position"] == [64.0, 65.0, 66.0]
    assert recorder.frames[0]["goal_relative_position"] == [67.0, 68.0, 69.0]
    assert recorder.frames[0]["extras_log"]["task_progress"] == 0.3


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return _FakeTensor(self.value[index])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value

    def item(self):
        return self.value

    def numel(self):
        return len(self.value)

    def sum(self):
        return _FakeTensor(sum(self.value))

    def __ge__(self, other):
        return _FakeTensor([1 if value >= other else 0 for value in self.value])
