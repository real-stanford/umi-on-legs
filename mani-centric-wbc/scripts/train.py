import hydra
from isaacgym import gymapi, gymutil  # must be improved before torch
from omegaconf import OmegaConf
from utils import setup

import wandb
from legged_gym.rsl_rl.runners.on_policy_runner import (
    OnPolicyRunner,
)


@hydra.main(config_path="../config", config_name="train", version_base="1.2")
def train(config):
    config_dict = OmegaConf.to_container(config, resolve=True)
    setup(config_dict, seed=config.seed)  # type: ignore
    sim_params = gymapi.SimParams()
    gymutil.parse_sim_config(config.env.cfg.sim, sim_params)
    env = hydra.utils.instantiate(config.env, sim_params=sim_params)
    config.runner.ckpt_dir = wandb.run.dir

    runner: OnPolicyRunner = hydra.utils.instantiate(
        config.runner, env=env, eval_fn=None
    )
    if config.ckpt_path is not None:
        runner.load(config.ckpt_path, load_optimizer=True)
    runner.learn()


if __name__ == "__main__":
    train()
