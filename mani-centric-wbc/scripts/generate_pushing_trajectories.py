import numpy as np
import pytorch3d.transforms as p3d
import torch
from transforms3d import euler
import pickle

if __name__ == "__main__":
    time_scaling = 1.0
    sampling_rate = 200
    parsed_plan = []
    distance = [1.0, 15.0]
    episode_len = 2000
    base_ee_rotation = euler.euler2mat(-1.57079632679, 0.0, -1.57079632679, axes="sxyz")
    extra_rotation = euler.euler2mat(0.0, 0.0, 0.0, axes="sxyz")
    rs = np.random.RandomState(0)
    for i in range(200):
        ee_pos = np.zeros((episode_len, 3))
        ee_pos[:, 2] = 0.05
        ee_pos[:, 0] = np.linspace(
            0.0, rs.uniform(distance[0], distance[1]), episode_len
        )
        t = np.linspace(0, len(ee_pos) / sampling_rate, len(ee_pos) + 1)
        rot_mat = base_ee_rotation @ euler.euler2mat(
            rs.uniform(-0.8, 0.0),
            rs.uniform(-0.05, 0.05),
            rs.uniform(-0.05, 0.05),
            axes="sxyz",
        )

        ee_axis_angle = p3d.matrix_to_axis_angle(torch.tensor(rot_mat)).numpy()
        ee_axis_angle = np.repeat(ee_axis_angle.reshape(1, 3), episode_len, axis=0)
        gripper_width = np.zeros_like(ee_axis_angle)

        parsed_plan.append(
            {
                "t": t,
                "ee_pos": ee_pos,
                "ee_axis_angle": ee_axis_angle,
                "gripper_width": gripper_width,
            }
        )
    pickle.dump(
        parsed_plan,
        open("longer_pushing_trajs.pkl", "wb"),
    )
