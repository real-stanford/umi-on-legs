# Copyright © 2018 Naturalpoint
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.

from typing import Dict, List, cast
import numpy as np
import sys
import time
from modules.mocap.natnet_client import NatNetClient
import modules.mocap.mocap_data as MoCapData

from rclpy.node import Node
import rclpy
from geometry_msgs.msg import PoseStamped

from functools import partial

import rclpy.publisher
from transforms3d import quaternions, affines


class MocapNode(Node):
    def __init__(
        self,
        rigid_body_dict: Dict[int, str],
        ip: str,
        use_multicast=True,
    ):
        super().__init__("mocap_node")

        self.ros2_publishers = {}
        for rigid_body_name in rigid_body_dict.values():
            self.ros2_publishers[rigid_body_name] = self.create_publisher(
                PoseStamped, f"mocap/{rigid_body_name}", 1
            )
        self.prev_receive_time = time.monotonic()

        streaming_client = NatNetClient()
        streaming_client.set_client_address("127.0.0.1")
        streaming_client.set_server_address(ip)
        streaming_client.set_use_multicast(use_multicast)

        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        streaming_client.new_frame_listener = partial(
            self.receive_new_frame,
            rigid_body_dict = rigid_body_dict,
            ros2_publishers=self.ros2_publishers,
        )

        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        is_running = streaming_client.run()
        if not is_running:
            print("ERROR: Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                print("exiting 1")

        time.sleep(1)
        if streaming_client.connected() is False:
            print(
                "ERROR: Could not connect properly.  Check that Motive streaming is on."
            )
            try:
                sys.exit(2)
            except SystemExit:
                print("...")
            finally:
                print("exiting 2")

        print("init done")
        

    def receive_new_frame(
        self,
        data_dict,
        rigid_body_dict: Dict[int, str],
        ros2_publishers: Dict[str, rclpy.publisher.Publisher],
    ):
        # print(f"receive_new_frame: {data_dict}")
        model_names = []
        marker_data_list = data_dict["marker_set_data"].marker_data_list
        for marker_data in marker_data_list:
            model_name = marker_data.model_name.decode("utf-8")
            if model_name != "all":
                model_names.append(model_name)


        rigid_body_list = data_dict["rigid_body_data"].rigid_body_list
        rigid_body_list = cast(List[MoCapData.RigidBody], rigid_body_list)


        for i, rigid_body in enumerate(rigid_body_list):
            if rigid_body.id_num not in rigid_body_dict:
                continue
            rigid_body_name = rigid_body_dict[rigid_body.id_num]
            name = rigid_body_name

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            mocap_robot_in_world_frame = affines.compose(
                T=rigid_body.pos,
                R=quaternions.quat2mat(
                    np.array(rigid_body.rot)[[3, 0, 1, 2]]
                ),  # rigid_body.rot is xyzw, need to convert to wxyz
                Z=np.ones(3),
            )
                
            trans, rotm, _, _ = affines.decompose(mocap_robot_in_world_frame)
            quat_wxyz = quaternions.mat2quat(rotm)

            pose_msg.pose.position.x = trans[0]
            pose_msg.pose.position.y = trans[1]
            pose_msg.pose.position.z = trans[2]

            pose_msg.pose.orientation.w = quat_wxyz[0]
            pose_msg.pose.orientation.x = quat_wxyz[1]
            pose_msg.pose.orientation.y = quat_wxyz[2]
            pose_msg.pose.orientation.z = quat_wxyz[3]
            ros2_publishers[name].publish(pose_msg)
        self.prev_receive_time = time.monotonic()
