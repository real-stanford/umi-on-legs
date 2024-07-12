import os
import bpy
import pickle
import numpy as np
from mathutils import Quaternion, Matrix, Vector, Euler

link_mapper = [
    "base",
    "FL_foot_target",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FL_foot",
    "FR_foot_target",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FR_foot",
    "Head_upper",
    "Head_lower",
    "RL_foot_target",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RL_foot",
    "RR_foot_target",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RR_foot",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "end_effector",
]


def import_animation(pickle_path: str):
    anim_data = pickle.load(open(pickle_path, "rb"))
    link_pos = np.array(anim_data["state_logs"]["rigid_body_pos"])
    link_quat_xyzw = np.array(anim_data["state_logs"]["rigid_body_xyzw_quat"])
    # only one actor supported
    assert link_pos.shape[1] == 1
    assert link_quat_xyzw.shape[1] == 1
    link_pos = link_pos[:, 0, :]
    link_quat_xyzw = link_quat_xyzw[:, 0, :]
    # swap time and link axis
    link_pos = np.swapaxes(link_pos, 0, 1)
    link_quat_xyzw = np.swapaxes(link_quat_xyzw, 0, 1)

    # isaac gym is Y-up Right-handed coordinate system
    # blender is  Z-up left-handed coordinate system
    # so we need to convert the quaternion from isaac gym to blender
    yup_to_zup = Euler((np.pi / 2, 0, 0), "XYZ").to_quaternion()
    for link_idx, (link_pos_traj, link_quat_traj) in enumerate(
        zip(link_pos, link_quat_xyzw)
    ):
        link_name = link_mapper[link_idx]
        if link_name not in bpy.data.objects:
            continue
        blender_obj = bpy.data.objects[link_name]
        for frame_idx, (pos, quat) in enumerate(zip(link_pos_traj, link_quat_traj)):
            # insert position keyframe
            blender_obj.location = pos
            blender_obj.keyframe_insert("location", frame=frame_idx)
            # change rotation mode to quaternion
            blender_obj.rotation_mode = "QUATERNION"
            # blender's quaternion constructor is in wxyz format
            # insert quaternion keyframe, applying transform
            blender_obj.rotation_quaternion = (
                Quaternion((quat[3], quat[0], quat[1], quat[2])) @ yup_to_zup
            )
            blender_obj.keyframe_insert("rotation_quaternion", frame=frame_idx)
    task = anim_data["task_logs"]
    axes = [bpy.data.objects[f"axes.{i:03d}"] for i in range(8)]
    print(axes)
    for frame_idx, (pos, quat) in enumerate(
        zip(task["target_positions"], task["target_quats_wxyz"])
    ):
        assert pos.shape[0] == len(axes)
        for ax, ax_pos, ax_quat_wxyz in zip(axes, pos, quat):
            ax.location = ax_pos
            ax.keyframe_insert("location", frame=frame_idx)
            ax.rotation_mode = "QUATERNION"
            ax.rotation_quaternion = Quaternion(ax_quat_wxyz)
            ax.keyframe_insert("rotation_quaternion", frame=frame_idx)


if __name__ == "__main__":
    pickle_path = "/path/to/run/dir/logs.pkl"
    import_animation(pickle_path)
