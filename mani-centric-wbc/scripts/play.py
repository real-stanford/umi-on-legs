import os
import pickle
import re
from isaacgym import gymapi, gymutil  # must be improved before torch
from argparse import ArgumentParser

import hydra
import imageio.v2 as imageio
import numpy as np
import zarr
import torch
from omegaconf import OmegaConf
from rich.progress import track
from transforms3d import affines, quaternions
from legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb
from legged_gym.env.isaacgym.env import IsaacGymEnv
from train import setup


def recursively_replace_device(obj, device: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "device":
                obj[k] = device
            else:
                obj[k] = recursively_replace_device(v, device)
        return obj
    elif isinstance(obj, list):
        return [recursively_replace_device(v, device) for v in obj]
    else:
        return obj
    return obj


count = 0


def play():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trajectory_file_path", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    args = parser.parse_args()
    if args.visualize:
        args.num_envs = 1

    config = OmegaConf.create(
        pickle.load(
            open(os.path.join(os.path.dirname(args.ckpt_path), "config.pkl"), "rb")
        )
    )
    sim_params = gymapi.SimParams()
    gymutil.parse_sim_config(config.env.cfg.sim, sim_params)
    config = recursively_replace_device(
        OmegaConf.to_container(
            config,
            resolve=True,
        ),
        device=args.device,
    )
    config["_convert_"] = "all"
    config["wandb"]["mode"] = "offline"  # type: ignore
    config["env"]["headless"] = not args.visualize  # type: ignore
    config["env"]["graphics_device_id"] = int(args.device.split("cuda:")[-1]) if "cuda" in args.device else 0  # type: ignore
    config["env"]["attach_camera"] = args.visualize  # type: ignore
    config["env"]["sim_device"] = args.device
    config["env"]["dof_pos_reset_range_scale"] = 0
    config["env"]["controller"]["num_envs"] = args.num_envs  # type: ignore
    config["env"]["cfg"]["env"]["num_envs"] = args.num_envs  # type: ignore
    config["env"]["controller"]["num_envs"] = args.num_envs  # type: ignore
    config["env"]["cfg"]["domain_rand"]["push_robots"] = False  # type: ignore
    config["env"]["cfg"]["domain_rand"]["transport_robots"] = False  # type: ignore

    # reset episode before commands change
    config["env"]["cfg"]["terrain"]["mode"] = "plane"
    config["env"]["cfg"]["init_state"]["pos_noise"] = [0.0, 0.0, 0.0]
    config["env"]["cfg"]["init_state"]["euler_noise"] = [0.0, 0.0, 0.0]
    config["env"]["cfg"]["init_state"]["lin_vel_noise"] = [0.0, 0.0, 0.0]
    config["env"]["cfg"]["init_state"]["ang_vel_noise"] = [0.0, 0.0, 0.0]
    config["env"]["tasks"]["reaching"]["sequence_sampler"][
        "file_path"
    ] = args.trajectory_file_path

    config["env"]["constraints"] = {}

    setup(config, seed=config["seed"])  # type: ignore

    env: IsaacGymEnv = hydra.utils.instantiate(
        config["env"],
        sim_params=sim_params,
    )
    config["runner"]["ckpt_dir"] = wandb.run.dir
    runner: OnPolicyRunner = hydra.utils.instantiate(
        config["runner"], env=env, eval_fn=None
    )
    runner.load(args.ckpt_path)
    policy = runner.alg.get_inference_policy(device=env.device)
    actor_idx: int = config["env"]["cfg"]["env"]["num_envs"] // 2

    def update_cam_pos():
        cam_rotating_frequency: float = 0.025
        offset = np.array([0.8, 0.3, 0.3]) * 1.5
        target_position = env.state.root_pos[actor_idx, :]
        # rotate camera around target's z axis
        angle = np.sin(2 * np.pi * env.gym_dt * cam_rotating_frequency * count)
        target_transform = affines.compose(
            T=target_position.cpu().numpy(),
            R=np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            ),
            Z=np.ones(3),
        )

        camera_transform = target_transform @ affines.compose(
            T=offset,
            R=np.identity(3),
            Z=np.ones(3),
        )
        try:
            camera_position = affines.decompose(camera_transform)[0]
            env.set_camera(camera_position, target_position)
        except np.linalg.LinAlgError:
            pass
        finally:
            pass

    obs, privileged_obs = env.reset()

    if args.visualize:
        env.render()  # render once to initialize viewer

    if args.num_steps == -1:
        with torch.inference_mode():
            while True:
                actions = policy(obs)
                obs = env.step(actions)[0]
                update_cam_pos()
                env.render()

    state_logs = {
        "root_state": [],
        "root_pos": [],
        "root_xyzw_quat": [],
        "root_lin_vel": [],
        "root_ang_vel": [],
        "rigid_body_pos": [],
        "rigid_body_xyzw_quat": [],
        "dof_pos": [],
        "dof_vel": [],
        "contact_forces": [],
        "episode_time": [],
    }
    action_logs = {
        "torque": [],
        "action": [],
        "obs": [],
    }
    task_logs = {
        "target_positions": [],
        "target_quats_wxyz": [],
    }
    episode_logs = {}
    task = list(env.tasks.values())[0]
    with imageio.get_writer(
        uri=os.path.join(wandb.run.dir, "video.mp4"), mode="I", fps=24
    ) as writer, torch.inference_mode():

        def render_cb(env, writer=writer):
            global count
            if env.state.time * 24 < count:
                return
            if args.visualize:
                update_cam_pos()
                env.visualize(vis_env_ids=[0])
                env.render()
                count += 1
                if args.record_video:
                    env.gym.write_viewer_image_to_file(
                        env.viewer, f"/{wandb.run.dir}/out.png"
                    )
                    img = imageio.imread(f"/{wandb.run.dir}/out.png")
                    writer.append_data(img)
            for k, v in state_logs.items():
                v.append(getattr(env.state, k)[:].cpu().numpy())
            action_logs["torque"].append(env.ctrl.torque[:].cpu().numpy())
            action_logs["action"].append(actions.view(args.num_envs, -1).cpu().numpy())
            action_logs["obs"].append(obs.view(args.num_envs, -1).cpu().numpy())
            global_target_pose = torch.stack(
                [
                    task.get_target_pose(
                        times=env.state.episode_time + t_offset,
                        sim_dt=env.state.sim_dt,
                    )
                    # NOTE: only for visualization purposes,
                    # you can change the target pose times to any
                    # time interval which visualizes the gripper
                    # movements best
                    for t_offset in np.linspace(0.05, 1.0, 8)
                ],
                dim=1,
            )
            task_logs["target_positions"].append(
                global_target_pose[..., :3, 3].cpu().squeeze().numpy()
            )
            task_logs["target_quats_wxyz"].append(
                np.vstack(
                    [
                        quaternions.mat2quat(rot_mat)
                        for rot_mat in global_target_pose[..., :3, :3]
                        .cpu()
                        .squeeze()
                        .numpy()
                    ]
                )
            )

        for step_idx in track(range(args.num_steps), description="Playing"):
            actions = policy(obs)
            obs, privileged_obs, rews, dones, infos = env.step(
                actions, callback=render_cb
            )
            for k, v in infos.items():
                episode_logs.setdefault(k, []).append(v[:].cpu().numpy())

            if args.visualize and dones[actor_idx]:
                break
    root = zarr.group(
        store=zarr.DirectoryStore(wandb.run.dir + "/logs.zarr"), overwrite=True
    )
    for k, v in {
        **state_logs,
        **action_logs,
        **episode_logs,
        **task_logs,
    }.items():
        v = np.array(v)
        k = k.replace("/", "_")
        if len(v.shape) == 2:
            root.create_dataset(
                k, data=v, chunks=(1, *list(v.shape)[1:]), dtype=v.dtype
            )
        else:
            root.create_dataset(k, data=v, dtype=v.dtype)
    pickle.dump(
        {
            "config": config,
            "state_logs": state_logs,
            "action_logs": action_logs,
            "episode_logs": episode_logs,
            "task_logs": task_logs,
        },
        open(wandb.run.dir + "/logs.pkl", "wb"),
    )


if __name__ == "__main__":
    play()
