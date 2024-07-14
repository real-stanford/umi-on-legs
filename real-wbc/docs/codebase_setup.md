# Real World Codebase Instructions

## Go2 Self-Calibration

Go2 runs a self-calibration process every time after it is powered on. Ensure you **squeeze all the legs** and let them **fully touch the ground** before turning on the battery. The whole-body controller will shake if we do not properly calibrate the Go2.
This **has** happened to us before, so make sure to triple-check that you've placed Go2 exactly in the zero position.
If the controller shakes, rebooting with a more careful calibration would be the first thing we recommend trying.

## Setup Dev Container

To improve reproducibility and make sure our workspace environment does not contaminate the Jetson environment, we develop all of our real-world code inside [VSCode Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers). It is a VSCode plugin that runs a docker container and mounts your workspace onto the container. Therefore, you can enjoy the docker environment while editing your code natively.

Before building the container, please ensure your Internet connection on your Go2 is fast enough (at least 1M/s). If not, you can follow the [network setup tutorial](network.md) to set up the Ethernet and WiFi connection.

To build the docker image, first, clone the repository into your **Go2's Jetson** and open a remote SSH VSCode workspace in `real-WBC`:
"`sh
# In Go2's Jetson
git clone https://github.com/real-stanford/umi-on-legs.git
cd umi-on-legs
git submodules update --init --recursive
code real-WBC
```

If you are **not** using the default `unitree` account in your Jetson or you want to run the controller on other devices, please check your user's `UID` and `GID` by typing terminal command `id`, and update `"USER_UID"` and `"USER_GID"`in `./devcontainer/devcontainer.json` accordingly. This step ensures that the file permission in the container is the same as your current user.

Then install VSCode plugin `Dev Containers`. After pressing `Ctrl-Shift-P`, search and select `Dev Containers: Rebuild and Reopen in Container`. The docker container should start to build. Building the docker image from scratch usually takes a long time (~30 min).

We customized the devcontainer with the following features:
1. `zsh` and several helpful autocompletion plugins. If you are not a `zsh` user, you need to **manually create a `.zsh_history` file in your home directory**, so it can be mounted to the container and keep all your history commands. Plugin [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions) allows you to press **right arrow key** to complete the line according to command history; [zsh-autocomplete](https://github.com/marlonrichert/zsh-autocomplete) helps you to complete commands using `tab` key and search history command through `Ctlr-R`.  
2. Using root in the container is generally not recommended. We create a user named `real` with `sudo` access for a better development experience (password is `real`), although having `sudo` access is also not recommended. To match the file permission, please make sure you have set the `USER_UID` and `USER_GID` arguments correctly according to your host machine user.
3. We added `"--network=host" ` in `.devcontainer/devcontainer.json` and installed Unitree-specific ROS2 (using cyclonedds). Therefore, the ROS2 environment inside the docker can communicate with Go2's controller board. Type `ros2 topic list` in the container terminal to check whether it has been set up successfully. If you are running the dev container elsewhere other than Go2's Jetson, you can check the network interface name that connects to Go2's LAN (under `192.168.123.xxx` segment) through` ip a' and append the following line to your Dockerfile
    ```
 RUN sed -i "s/eth0/network_name/g" \
 /home/${USERNAME}/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml
    ```
    `network_name` should be replaced by your actual network interface name.

## Disable Sports Mode

By default, Go2 is set to sports mode with high-level control by a joystick. If you call Go2's low-level motor API, it will conflict with the internal high-level commands and shake heavily. Therefore, you must lay the dog on the ground and disable its sports mode. If you don't want to disable it through your mobile phone, an easier way is through `unitree_sdk2`. 

We wrote a script `disable_sports_mode_go2.cpp` script using `unitree_sdk2` (available [here](https://github.com/yihuai-gao/unitree_sdk2)) which is installed inside the container. After turning on the container, run 
"`sh
~/unitree_sdk2/build/disable_sports_mode_go2 eth0
```
To disable the sports mode immediately. If you are using other devices, change `eth0` to the correct network interface name (under `192.168.123.xxx` segment).

[Our modified sdk](https://github.com/yihuai-gao/unitree_sdk2) also provides a python `crc_module`. It calculates the CRC value for each low-level ROS2 message before sending it to Go2's controller board.  

## Run Pose Estimator

We support the iPhone and OptiTrack motion capture systems as robot pose estimators. iPhone is more accessible and works well in the wild, but it has a longer delay and is less stable than MoCap, especially in dynamic actions. You only need to run one of the two methods for the whole-body controller.

### iPhone

- Build and install the [iPhoneVIO iOS app](https://github.com/yihuai-gao/iPhoneVIO) to your iPhone. 
- Connect the USB-Ethernet adapter to the iPhone and use an ethernet cable to connect to the ethernet port on Go2's Jetson. The port on the USB extension dock does not work in this case.
- Manually set the IP address on your iPhone (e.g., ip: `192.168.123.170`, netmask: `255.255.255.0`, gateway `192.168.123.1`) for this ethernet connection. You can use numbers other than `170` if no other existing devices occupy it.
- Run the script on Go2's Jetson.
    "`sh
    python scripts/run_pose_estimator.py
    ```
- Open the iPhone app and press the refresh button. If you see the IP address on the iPhone, it is successfully connected.
### MoCap
After setting up the markers on the dog and the `Motive` software as the server. Get the ip address of the server and run
"`sh
python scripts/run_mocap_node.py your_ip
```
If it is not connected, check the dog's network connection.

## Setup ARX5 SDK

When running devcontainer, `arx5-sdk` should be already mounted in path `../arx5-sdk`. You can either build the python binding library inside or outside the container.

Please follow README in the [github repository](https://github.com/yihuai-gao/arx5-sdk) to set up the robot arm. You can run some test scripts to check the functionality inside the container.


## SpaceMouse Tele-Operation

First, install relative libraries and enable the space mouse service in the **host machine (outside devcontainer)**:

"`sh
sudo apt install libspnav-dev spacenavd
sudo systemctl enable spacenavd.service
sudo systemctl start spacenavd.service
```
It will create a socket `/var/run/spnav.sock`, and we will mount it to the container. The first time you start the services, you need to restart the container to ensure it is connected.


## Run Whole Body Controller

After checking all the other components (sports mode disabled, mocap/iphone connected, arx5 connected), you can start to run the whole-body controller
"`sh
python scripts/run_wbc.py --ckpt_path your_checkpoint.pt --pickle_path your_trajectory.pkl --traj_idx 0 --pose_estimator iphone --use_realtime_target  --fix_at_init_pose
```
Argument `--pose_estimator` depends on the pose estimator you are using (iphone or mocap). `--use_realtime_target` will let the controller listen to external trajectory updates. `--fix_at_init_pose` will disable the trajectory replay.

Joystick key mapping: 
- L1: Emergency stop. Ensure you can always push this button in case of any dangerous behaviors.
- R1: The robot gradually stands up (interpolating to the target joint actions)  
- L2: Start RL controller. Before pressing this button, please hold the arm tightly if it shakes badly. The robot should be able to stabilize itself in a few seconds.
- R2: Stop RL controller. The controller will stop running, and the motor command will be fixed to the last output action. Press L2 to restart the controller.

After starting the RL controller, you may disturb the robot and gripper to check whether it is tracking poses well in the task space. The gripper should stay in the same spot when you drag the dog's body.

Finally, for space mouse teleoperation, run
"`sh
python scripts/run_teleop.py
```
and you should be able to control the gripper pose. Notice that the coordinate frame of the iPhone or mocap can be different from the space mouse. You need to find out the correct orientation through trial and error.

After the RL controller stops, you should manually kill the `run_teleop.py` script and run this script after the RL controller starts again.


## Autonomous Diffusion Policy

The above instructions is sufficient to reproduce the whole-body controller with basic task-space tele-operation tracking.
The code for deploying diffusion policy with WBCs will be released in a month or two.
As mentioned in our [hardware reflections](./hardware_design_choices.md), our Go2 broke recently.
Therefore, this part of the code has not been tested and is not ready for release.
In the mean time, for untested code, feel free to email us.