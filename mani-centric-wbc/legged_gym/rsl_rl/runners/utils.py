from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
from legged_gym.rsl_rl.storage.rollout_storage import RolloutStorage


def parse_rollout_stats(
    storage: RolloutStorage,
    cum_rollout_stats: Dict[str, torch.Tensor],
    final_rollout_stats: Dict[str, Deque[float]],
    vis_frames: List[np.ndarray],
    gamma: float,
    vis_resolution: Tuple[int, int],
) -> Dict[str, float]:
    """
    Parses rollout statistics from the rollout storage. To record returns,
    episode length, etc.
    """
    for rewards, dones, infos in zip(storage.rewards, storage.dones, storage.infos):
        # rewards.shape = (num_envs, )
        # dones.shape = (num_envs, )
        # infos : Dict[str, (num_envs, )]
        cum_rollout_stats["reward"] += rewards.squeeze(dim=1)
        cum_rollout_stats["episode_len"] += 1
        cum_rollout_stats["return"] += rewards.squeeze(dim=1) * (
            gamma ** cum_rollout_stats["episode_len"]
        )

        for k, v in infos.items():
            if v.shape == rewards.squeeze(dim=1).shape:
                if k not in cum_rollout_stats.keys():
                    cum_rollout_stats[k] = torch.zeros_like(v)
                cum_rollout_stats[k] += v
            elif v.shape == (
                *vis_resolution,
                3,
            ):
                # vis frame
                vis_frames.append(v)
            else:
                raise RuntimeError(
                    f"infos[{k}].shape = {v.shape} != rewards.shape = {rewards.squeeze(dim=1).shape}"
                )
        if not dones.any():
            continue
        new_ids = dones[:, 0].nonzero(as_tuple=False)
        keys = set(cum_rollout_stats.keys())
        keys.remove("episode_len")
        for k in list(keys) + ["episode_len"]:  # do episode len last
            sum_key = f"{k}/sum"
            mean_key = f"{k}/mean"
            if sum_key not in final_rollout_stats.keys():
                deque_length = final_rollout_stats["reward"].maxlen
                final_rollout_stats[sum_key] = deque(maxlen=deque_length)
            if mean_key not in final_rollout_stats.keys():
                deque_length = final_rollout_stats["reward"].maxlen
                final_rollout_stats[mean_key] = deque(maxlen=deque_length)

            values = (
                cum_rollout_stats[k][new_ids]
                .squeeze(dim=1)
                .float()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            final_rollout_stats[sum_key].extend(values)
            final_rollout_stats[mean_key].extend(
                values
                / cum_rollout_stats["episode_len"][new_ids]
                .squeeze(dim=1)
                .cpu()
                .numpy()
                .reshape(-1)
            )
            cum_rollout_stats[k][new_ids] = 0.0

    return {k: sum(v) / len(v) for k, v in final_rollout_stats.items() if len(v) > 0}
