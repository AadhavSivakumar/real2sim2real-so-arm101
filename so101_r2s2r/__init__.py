"""SO-101 cube-into-cup real2sim2real tasks (gym registration)."""
import gymnasium as gym

gym.register(
    id="SO101-CubeInCup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:CubeInCupEnvCfg"},
    disable_env_checker=True,
)
gym.register(
    id="SO101-CubeInCup-Cam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:CubeInCupCamEnvCfg"},
    disable_env_checker=True,
)
gym.register(
    id="SO101-CubeInCup-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:CubeInCupRLEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CubeInCupPPORunnerCfg"},
    disable_env_checker=True,
)


def register_tasks():
    """``--external_callback so101_r2s2r.register_tasks`` hook for IsaacLab's train/play CLIs
    (registration already happened on import; returning None keeps all CLI overrides)."""
    return None
