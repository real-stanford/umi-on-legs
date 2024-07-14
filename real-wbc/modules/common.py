from __future__ import annotations
from enum import IntEnum
from typing import Dict, List, Type

import numpy as np


class MotorId(IntEnum):
    FR_HIP = 0
    FR_THIGH = 1
    FR_CALF = 2
    FL_HIP = 3
    FL_THIGH = 4
    FL_CALF = 5
    RR_HIP = 6
    RR_THIGH = 7
    RR_CALF = 8
    RL_HIP = 9
    RL_THIGH = 10
    RL_CALF = 11

    @classmethod
    def values(cls: Type[MotorId]):
        return [i.value for i in cls]

    @classmethod
    def keys(cls: Type[MotorId]):
        return [i.name for i in cls]


POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
LEG_DOF = 12
SDK_DOF = 20


interface_joint_order = [
    MotorId.FR_HIP,
    MotorId.FR_THIGH,
    MotorId.FR_CALF,
    MotorId.FL_HIP,
    MotorId.FL_THIGH,
    MotorId.FL_CALF,
    MotorId.RR_HIP,
    MotorId.RR_THIGH,
    MotorId.RR_CALF,
    MotorId.RL_HIP,
    MotorId.RL_THIGH,
    MotorId.RL_CALF,
]
policy_joint_order = [
    MotorId.FL_HIP,
    MotorId.FL_THIGH,
    MotorId.FL_CALF,
    MotorId.FR_HIP,
    MotorId.FR_THIGH,
    MotorId.FR_CALF,
    MotorId.RL_HIP,
    MotorId.RL_THIGH,
    MotorId.RL_CALF,
    MotorId.RR_HIP,
    MotorId.RR_THIGH,
    MotorId.RR_CALF,
]

torque_limits: Dict[int, float] = {
    MotorId.FR_HIP: 40,
    MotorId.FR_THIGH: 60,
    MotorId.FR_CALF: 75,
    MotorId.FL_HIP: 40,
    MotorId.FL_THIGH: 60,
    MotorId.FL_CALF: 75,
    MotorId.RR_HIP: 40,
    MotorId.RR_THIGH: 60,
    MotorId.RR_CALF: 75,
    MotorId.RL_HIP: 40,
    MotorId.RL_THIGH: 60,
    MotorId.RL_CALF: 75,
}


def rematch_joint_order(
    prev_joint_order: List[MotorId],
    new_joint_order: List[MotorId],
    prev_val: npt.NDArray[np.float64],
):
    assert len(prev_joint_order) == len(new_joint_order)
    # assert prev_val.shape == (len(prev_joint_order),)
    if len(prev_val) == len(prev_joint_order):
        new_val = np.zeros_like(prev_val)
        for i, j in enumerate(new_joint_order):
            new_val[i] = prev_val[prev_joint_order.index(j)]
        return new_val
    else:
        return prev_val


def reorder(prev_val):
    return rematch_joint_order(
        interface_joint_order,
        policy_joint_order,
        prev_val,
    )
