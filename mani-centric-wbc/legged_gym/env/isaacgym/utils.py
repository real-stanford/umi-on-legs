from typing import Union

import numpy as np
import pytorch3d.transforms as pt3d
import torch
from isaacgym.torch_utils import quat_apply, normalize


def torch_rand_float(
    lower: Union[float, torch.Tensor],
    upper: Union[float, torch.Tensor],
    shape: tuple,
    device: str,
    generator: torch.Generator,
):
    return (upper - lower) * torch.rand(
        *shape, device=device, generator=generator
    ) + lower


def orn_diff_matrix(
    matrix_1: torch.Tensor,
    matrix_2: torch.Tensor,
) -> torch.Tensor:
    if torch.allclose(matrix_1, matrix_2):
        return torch.zeros(matrix_1.shape[0], device=matrix_1.device)
    assert matrix_1.shape == matrix_2.shape
    assert (
        not torch.isnan(matrix_1).any() and not torch.isnan(matrix_2).any()
    ), f"matrix_1: {matrix_1}, matrix_2: {matrix_2}"
    rot_mat = matrix_1 @ matrix_2.transpose(1, 2)
    orn_dist = torch.arccos((rot_mat.diagonal(dim1=1, dim2=2).sum(dim=1) - 1) / 2) % (
        2 * np.pi
    )
    return torch.min(
        orn_dist,
        2 * np.pi - orn_dist,
    )


def orn_diff_axis_angles(
    axis_angle_1: torch.Tensor,
    axis_angle_2: torch.Tensor,
) -> torch.Tensor:
    if torch.allclose(axis_angle_1, axis_angle_2):
        return torch.zeros(axis_angle_1.shape[0], device=axis_angle_1.device)
    assert axis_angle_1.shape == axis_angle_2.shape
    assert (
        not torch.isnan(axis_angle_1).any() and not torch.isnan(axis_angle_2).any()
    ), f"axis_angle_1: {axis_angle_1}, axis_angle_2: {axis_angle_2}"
    rot_mat = pt3d.axis_angle_to_matrix(axis_angle_1) @ pt3d.axis_angle_to_matrix(
        axis_angle_2
    ).transpose(1, 2)
    orn_dist = pt3d.matrix_to_axis_angle(rot_mat).norm(dim=1) % (2 * np.pi)
    return torch.min(
        orn_dist,
        2 * np.pi - orn_dist,
    )


def vec_to_forward_rotmat(vec: torch.Tensor) -> torch.Tensor:
    assert len(vec.shape) >= 2 and vec.shape[-1] == 3
    nvec = vec / vec.norm(dim=-1, keepdim=True)
    forward = nvec
    u = torch.zeros_like(nvec)
    u[..., 2] = 1
    mask = (nvec.abs()[..., 2] - torch.ones_like(nvec[..., 2])).abs() < 1e-4
    # change up vector
    u[mask] = torch.tensor([0.0, 1.0, 0.0], device=vec.device)
    s = torch.cross(forward, u)
    s = s / s.norm(dim=-1, keepdim=True)
    u = torch.cross(s, forward)
    return torch.cat(
        [
            s,
            u,
            -forward,
        ],
        dim=-2,
    ).reshape(*list(vec.shape)[:-1], 3, 3)


def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)
