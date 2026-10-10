"""PPO (rsl_rl) config for the privileged-state cube-into-cup task (adapted from IsaacLab's Franka lift)."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class FlooredGaussianCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian with a floor on the action std (run 3 collapsed to 0.03 on some joints and stopped
    exploring). Isaac Lab's cfg lacks rsl_rl's std_range; rsl_rl forwards every key to the distribution."""

    std_range: tuple = (0.15, 1.5)


@configclass
class CubeInCupPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "so101_cube_in_cup"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True,
                             distribution_cfg=FlooredGaussianCfg(init_std=1.0))
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.006,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
        gamma=0.98, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )
