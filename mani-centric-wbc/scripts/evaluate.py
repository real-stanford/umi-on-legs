import hydra
from isaacgym import gymapi, gymutil  # must be improved before torch
from omegaconf import OmegaConf
from utils import setup
import wandb
from legged_gym.rsl_rl.runners.on_policy_runner import (
    OnPolicyRunner,
)


@hydra.main(config_path="../config", config_name="eval", version_base="1.2")
def evaluate(config):
    config_dict = OmegaConf.to_container(config, resolve=True)
    setup(config_dict, seed=config.seed)  # type: ignore
    sim_params = gymapi.SimParams()
    gymutil.parse_sim_config(config.env.cfg.sim, sim_params)
    env = hydra.utils.instantiate(config.env, sim_params=sim_params)
    config.runner.ckpt_dir = wandb.run.dir
    runner: OnPolicyRunner = hydra.utils.instantiate(
        config.runner, env=env, eval_fn=None
    )
    assert config.ckpt_path is not None
    runner.load(config.ckpt_path, load_optimizer=True)

    # trial run through one step
    obs, privileged_obs = env.reset()
    action = runner.alg.act(obs, privileged_obs)
    obs, privileged_obs, rewards, dones, infos = env.step(
        action=action,
        return_vis=False,
    )
    privileged_obs = privileged_obs if privileged_obs is not None else obs
    runner.alg.process_env_step(rewards, dones, infos)

    eval_stats = runner.eval(use_pbar=True)
    wandb.log(eval_stats)


if __name__ == "__main__":
    evaluate()
