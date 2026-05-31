"""Unitree H1 tabletop transfer manipulation task."""

from __future__ import annotations

import gymnasium as gym

from . import agents


TASK_ID = "RoboGenesis-H1-Tabletop-Manipulation-v0"
TRANSFER_TASK_ID = "RoboGenesis-H1-Tabletop-Transfer-v0"


def _register(task_id: str) -> None:
    try:
        gym.spec(task_id)
        return
    except gym.error.Error:
        pass

    gym.register(
        id=task_id,
        entry_point=f"{__name__}.h1_tabletop_transfer_env:H1TabletopTransferEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.h1_tabletop_transfer_env:H1TabletopTransferEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1TabletopTransferPPORunnerCfg",
        },
    )


_register(TASK_ID)
_register(TRANSFER_TASK_ID)
