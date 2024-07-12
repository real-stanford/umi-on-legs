import typing
from typing import Any, List

import pydantic
import pytorch3d.transforms as pt3d
import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul

from legged_gym.env.isaacgym.state import EnvSetup, EnvState
from legged_gym.env.obs import Config, ObservationAttribute


@pydantic.dataclasses.dataclass(config=Config)
class EnvObservationAttribute(ObservationAttribute):
    # make sure `key` is an attribute of `EnvState`
    @pydantic.validator("key")
    def _check_key(cls, v):
        valid_key = hasattr(EnvState, v) or v in EnvState.__annotations__.keys()
        assert valid_key, f"key {v!r} is not an attribute of EnvState"
        return v


@pydantic.dataclasses.dataclass(config=Config)
class FeetContactAttribute(EnvObservationAttribute):
    feet_sensor_indices: List[int]
    force_threshold: float

    def __call__(self, struct: Any, generator: torch.Generator) -> torch.Tensor:
        force_sensor_tensor = super().__call__(struct=struct, generator=generator)
        return (
            force_sensor_tensor[:, self.feet_sensor_indices, 2].abs()
            > self.force_threshold
        )


@pydantic.dataclasses.dataclass(config=Config)
class RigidBodyPosAttribute(EnvObservationAttribute):
    rigid_body_idx: int
    local_frame: bool

    def __call__(self, struct: Any, generator: torch.Generator) -> torch.Tensor:
        all_pos = super().__call__(struct=struct, generator=generator)
        rigid_body_pos = all_pos[:, self.rigid_body_idx, :]
        if self.local_frame:
            state = typing.cast(EnvState, struct)
            # transform rigid body position to local frame
            rigid_body_pos = rigid_body_pos - state.root_pos
        return rigid_body_pos


@pydantic.dataclasses.dataclass(config=Config)
class RigidBodyOrientationAttribute(EnvObservationAttribute):
    rigid_body_idx: int
    local_frame: bool
    representation: str

    @pydantic.validator("representation")
    def _check_key(cls, v):
        assert v in [
            "quat_wxyz",
            "euler",
            "mat",
            "6dmat",
        ], f"representation {v!r} is not supported"
        return v

    def __call__(self, struct: Any, generator: torch.Generator) -> torch.Tensor:
        all_quat_xyzw = super().__call__(struct=struct, generator=generator)
        rigid_body_quat_xyzw = all_quat_xyzw[:, self.rigid_body_idx]
        if self.local_frame:
            state = typing.cast(EnvState, struct)
            # transform rigid body position to local frame
            rigid_body_quat_xyzw = quat_mul(
                quat_conjugate(state.root_xyzw_quat),
                rigid_body_quat_xyzw,
            )
        if self.representation == "quat_wxyz":
            return rigid_body_quat_xyzw[:, [3, 0, 1, 2]]
        # pt3d quaternion convention is wxyz
        rotmat = pt3d.quaternion_to_matrix(rigid_body_quat_xyzw[:, [3, 0, 1, 2]])
        if self.representation == "mat":
            return rotmat
        elif self.representation == "6dmat":
            # top two rows of rotation matrix
            return pt3d.matrix_to_rotation_6d(rotmat)
        elif self.representation == "euler":
            return pt3d.matrix_to_euler_angles(rotmat, "XYZ")
        else:
            raise NotImplementedError(f"representation {self.representation!r}")


@pydantic.dataclasses.dataclass(config=Config)
class EnvSetupAttribute(ObservationAttribute):
    # make sure `key` is an attribute of `EnvState`
    @pydantic.validator("key")
    def _check_key(cls, v):
        valid_key = hasattr(EnvSetup, v) or v in EnvSetup.__annotations__.keys()
        assert valid_key, f"key {v!r} is not an attribute of EnvSetup"
        return v
