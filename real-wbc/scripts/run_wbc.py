import datetime
import pytz
import zarr
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from typing import Dict, List, Optional
import typing
import logging
from rich.logging import RichHandler

from modules.common import (
    LEG_DOF,
    POS_STOP_F,
    SDK_DOF,
    VEL_STOP_F,
    MotorId,
    reorder,
    torque_limits,
)
import scipy.signal as signal
from transforms3d import affines, quaternions, euler, axangles

from modules.realtime_traj import RealtimeTraj

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crc_module import get_crc
from modules.velocity_estimator import MovingWindowFilter, VelocityEstimator
import numpy as np
import torch
import faulthandler

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    LowCmd,
    MotorCmd,
)
import time
import hydra
from omegaconf import OmegaConf
from geometry_msgs.msg import PoseStamped
from rclpy.time import Time

from robot_state.msg import EEFState, EEFTraj


def quat_rotate_inv(q: np.ndarray, v: np.ndarray):
    return quaternions.rotate_vector(
        v=v,
        q=quaternions.qinverse(q),
    )


from collections import deque
import time
import numpy as np
import os
import sys

sys.path.append("/home/real/arx5-sdk/python")
import arx5_interface as arx5


from modules.wbc_node import WBCNode


if __name__ == "__main__":

    np.set_printoptions(precision=3)
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    rclpy.init(args=None)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pickle_path", type=str, required=True)
    parser.add_argument("--traj_idx", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fix_at_init_pose", action="store_true")
    parser.add_argument("--use_realtime_target", action="store_true")
    parser.add_argument("--pose_estimator", type=str, default="iphone")
    args = parser.parse_args()
    wbc_node = WBCNode(**vars(args))
    logging.info("Deploy node ready")
    lowstate = wbc_node.arx5_joint_controller.get_state()
    if (lowstate.pos() == 0.0).all() and (lowstate.vel() == 0.0).all():
        logging.error("Arm is not connected!")
        exit(1)
    try:
        rclpy.spin(wbc_node)
    finally:
        rclpy.shutdown()
