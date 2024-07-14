# Real World Codebase Instructions

## Go2 Self-Calibration

Go2 runs a self-calibration process every time after powered on. Make sure you **squeeze all the legs** and let them **fully touch the ground** before turing on the battery. If Go2 is not properly calibrated, the whole-body controller may shake heavily.

## Setup Dev Container

To improve reproducibility and make sure our workspace environment does not contaminate the Jetson environment, we develop all of our real-world code inside [VSCode Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers). It is a VSCode plugin that runs a docker container and mounts your workspace onto the container. Therefore, you can enjoy the docker environment while editing your code natively.

Before building the container, please make sure your Internet connection on your Go2 is fast enough (at least 1M/s). If not, you can follow the [network setup tutorial](network.md) to setup the Ethernet and WiFi connection.

To build the docker image, first clone the repository into your **Go2's Jetson** and open a remote SSH VSCode workspace in `real-wbc`:
```sh
# In Go2's Jetson
git clone https://github.com/real-stanford/umi-on-legs.git
cd umi-on-legs
git submodules update --init --recursive
code real-wbc
```

If you are **not** using the default `unitree` account in your Jetson or you want to run the controller on other devices, please check your user's `UID` and `GID` by typing terminal command `id`, and update `"USER_UID"` and `"USER_GID"`in `./devcontainer/devcontainer.json` accordingly. This step makes sure that the file permission in the container is the same as your current user.

Then install VSCode plugin `Dev Containers`. After pressing `Ctrl-Shift-P`, search and select `Dev Containers: Rebuild and Reopen in Container`. The docker container should start to build. It usually take a long time (~30 min) to build the docker image from scratch.

We customized the devcontainer with the following features:
1. `zsh` and several helpful autocompletion plugins. If you are not a `zsh` user, you need to **manually create a `.zsh_history` file in your home directory**, so that it can be mounted to the container and keep all your history commands. Plugin [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions) allows you to press **right arrow key** to complete the line according to command history; [zsh-autocomplete](https://github.com/marlonrichert/zsh-autocomplete) helps you to complete commands using `tab` key and search history command through `Ctlr-R`.  
2. Using root in the container is generally not recommended. We create a user named `real` and with `sudo` access for better development experience (password is `real`), although having `sudo` access is not recommended as well. To match the file permission, please make sure you have set the `USER_UID` and `USER_GID` arguments correctly according to your host machine user.
3. We added `"--network=host"` in `.devcontainer/devcontainer.json` and installed Unitree specific ROS2 (using cyclonedds). Therefore the ROS2 environment inside the docker is able to communicate with Go2's controller board. Type `ros2 topic list` in the container terminal to check whether it is set up successfully. If you are running the devcontainer elsewhere other than Go2's Jetson, you can check the network interface name that connects to Go2's LAN (under `192.168.123.xxx` segment) though `ip a` and append the following line to your Dockerfile
    ```
    RUN sed -i "s/eth0/network_name/g"\
        /home/${USERNAME}/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml
    ```
    `network_name` should be replaced by your actual network interface name.

## Disable Sports Mode

By default, Go2 is set to sports mode with joystick high-level control. If you call its low-level motor API, it will conflict with the internal high-level commands and shake heavily. Therefore, you need to lay the dog on the ground and disable its sports mode. If you don't want to disable it through your mobile phone, a easier way is through `unitree_sdk2`. 

We wrote a script `disable_sports_mode_go2.cpp` script using `unitree_sdk2` (available [here](https://github.com/yihuai-gao/unitree_sdk2)) which is installed inside the container. After turning on the container, run 
```sh
~/unitree_sdk2/build/disable_sports_mode_go2 eth0
```
to disable the sports mode immediately. If you are using other devices, change `eth0` to the correct network interface name (under `192.168.123.xxx` segment).

[Our modified sdk](https://github.com/yihuai-gao/unitree_sdk2) also provides a python `crc_module`. It calculates the crc value for each low-level ROS2 message before sending it to Go2's controller board.  

## Run Pose Estimator

We support iPhone and OptiTrack motion capture system as robot pose estimators. iPhone is more accessible and works well in-the-wild, but it has longer delay and is less stable than MoCap especially in dynamic actions. You only need to run one of the two methods for the whole-body controller.

### iPhone

- Build and install the [iPhoneVIO iOS app](https://github.com/yihuai-gao/iPhoneVIO) to your iPhone. 
- Connect the USB-Ethernet adapter to iPhone and use an ethernet cable to connect to the ethernet port on Go2's Jetson. The port on the USB extension dock does not work in this case.
- Manually set IP address on your iPhone (e.g. ip: `192.168.123.170`, netmask: `255.255.255.0`, gateway `192.168.123.1`) for this ethernet connection. You can use other numbers than `170` as long as not it is not occupied by other existing devices.
- Run the script on Go2's Jetson.
    ```sh
    python scripts/run_pose_estimator.py
    ```
- Open the iPhone app and press the refresh button. If you see the ip address from the iPhone, it is successfully connected.
### MoCap
After setting up the markers on the dog and the `Motive` software as the server. Get the ip address of the server and simply run
```sh
python scripts/run_mocap_node.py your_ip
```
If it is not connected, check the network connection of the dog.

## Setup ARX5 SDK

When running devcontainer, `arx5-sdk` should be already mounted in path `../arx5-sdk`. You can either build the python binding library inside or outside the container.

Please follow README in the [github repository](https://github.com/yihuai-gao/arx5-sdk) to set up the robot arm. You can run some test scripts to check the functionality inside the container.


## SpaceMouse Tele-Operation

First install relative libraries and enable the spacemouse service in the **host machine (outside devcontainer)**:

```sh
sudo apt install libspnav-dev spacenavd
sudo systemctl enable spacenavd.service
sudo systemctl start spacenavd.service
```
It will create a socket `/var/run/spnav.sock` and we will mount it to the container. The first time you start the services, you need to restart the container to make sure it is connected.


## Run Whole Body Controller

After checking all the other components (sports mode disabled, mocap/iphone connected, arx5 connected), you can start to run the whole-body controller
```sh
python scripts/run_wbc.py --ckpt_path your_checkpoint.pt --pickle_path your_trajectory.pkl --traj_idx 0 --pose_estimator iphone --use_realtime_target  --fix_at_init_pose
```
Argument `--pose_estimator` depends on the pose estimator you are using (iphone or mocap). `--use_realtime_target` will let the controller listen to external trajectory updates. `--fix_at_init_pose` will disable the trajectory replay.

Joystick key mapping: 
- L1: Emergency stop. Make sure you can always push this button in case of any dangerous behaviors.
- R1: The robot gradually stands up (interpolating to the target joint actions)  
- L2: Start RL controller. Before pressing this button, please hold the arm tightly in case it shakes badly. The robot should be able to stabilize itself in a few seconds.
- R2: Stop RL controller. The controller will stop running and the motor command is fixed to the last output action. Press L2 to restart the controller.

After starting the RL controller, you may disturb robot and gripper to check whether it is tracking poses well in the task space. The gripper should stay in the same spot when you drag the body of the dog.

Finally, for spacemouse tele-operation, run
```sh
python scripts/run_teleop.py
```
and you should be able to control the gripper pose. Notice that the coordinate frame of iPhone or mocap can be different from the spacemouse. You need to find out the correct orientation through trial and error.

After the RL controller stopped, you should manually kill the `run_teleop.py` script and run this script after the RL controller starts again.


## Autonomous Diffusion Policy

Unfortunately, since our Go2 broke recently, this part of the code is not tested and not ready to release yet. We hope the above instructions are enough to reproduce the whole-body controller with basic task-space tele-operation tracking. Please stay tuned for our update!