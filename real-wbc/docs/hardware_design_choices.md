# Hardware Design Choices

During the development of our hardware system, we encountered many unexpected twist and turns. Here we summarize a few reflections that can be helpful for the community to build similar dog-with-arm robot systems.

1. Is Go2 a good choice for the base?
2. Is ARX5 a good choice for the arm? 
3. Is iPhone a good visual-inertial odometer (VIO)?
4. Is GoPro a good choice for data collection and policy deployment?

We also encourage more researchers to share your experience with the robotics community.

## Go2

In general, Go2 is robust, cheap and lightweight. Its hardware never break during our experiments. The only problem is that, after mounting a robot arm onto it and letting our controller stand for a while, the motors quickly heat up and trigger the emergency stop (the LED on Go2's turns red and we have to restart the entire robot). We use ice packs to cool it down before our next experiment. This is more frequent (5~10min) if the trained controller does not optimize the torque usage of its front legs. Although in the final evaluations, we trained controllers that can keep standing for more than 30 minutes, we still expect a better robot dog that has higher payload than Go2.

Since B2 is too big and heavy, a intermediate size like A1 or AlienGo is more ideal. We used AlienGo in the beginning but the hardware frequently broke (which took us 3 weeks waiting for the motor replacement every time). This is probably because the AlienGo model is too old (released in 2019) and the motors are not well protected. We would recommend a new generation of AlienGo or A1 from Unitree for future projects.

Update: Our Go2 broke when I was writing this documentation. There were no direct causes but the controller board inside Go2 (the mini PC that runs the low level communication, not the Jetson) cannot connect to any motors. We are waiting for the replacement thus the diffusion policy codebase is not ready for release yet.

## ARX5

We use ARX5 because its weight (3.4kg), reach range (620mm) and payload (1.5kg) fits our project well. To mount GoPro and fin-ray gripper on the end-effector, servo-based arms like ViperX may not provide enough payload (750g). On the other hand, Unitree Z1 (5kg) is too heavy to mount on Go2 and the gripper is not customizable for the UMI setup. Since ARX5 uses planetary gear motors, the precision is not as high as harmonic drive arms (e.g. UR5, Z1), but it is not a bottleneck for a mobile robot with a shakier base. We are grateful to ARX Technology for providing us the ARX5 arm.

Initially, we faced several challenges when using the arm - mainly due to software issues.
The original codebase is mostly used for the ALOHA with ROS1, where the space for customization is very limited. 
We developed a stand-alone [python sdk](https://github.com/yihuai-gao/arx5-sdk) supporting both joint and cartesian control. This sdk is open-sourced for customizability and is actively used in several ongoing projects. We only broke its motor once when we didn't correctly implement the current protection in the new sdk. The arm no longer break after we fixed it. 
We also re-calibrated the URDF model to get more precise inertia matrix (updated in our sdk). 

After these efforts, our usage of the arm is stable and robust. 

## iPhone

Our whole-body controller takes relative target end-effector pose as one of its inputs (i.e. target pose in task space relative to the current end-effector pose in task space, the latter requires robot pose estimation), therefore we need an accurate and robust robot pose estimation device. We use iPhone as a visual-inertial odometer (VIO) that can be mounted on Go2's back and provide real time odometry. We wrote a customized iOS app that keeps publishing camera poses from Apple ARKit to Go2's Jetson through ethernet. 

According to [this benchmark paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9785098/), iPhone provides robust tracking with the highest precision compared to Google ARCore, T265 and ZED2. Other options like [Vive Ultimate Tracker](https://www.vive.com/us/accessory/vive-ultimate-tracker/?wpsrc=Google%20AdWords&wpcid=21059541807&wpsnetn=x&wpkwn=&wpkmatch=&wpcrid=&wpscid=&wpkwid=&gad_source=1&gclid=CjwKCAjwy8i0BhAkEiwAdFaeGG06XJOZhQe5zNWUXiTxXNXLKZ6zyuDfabP8k9ZB4BGyIQo4RcVU1xoCqQUQAvD_BwE) have to go through a long map-building stage which we want to avoid. However, iPhone still has some limitations in our real-time control scenario:

- Although iPhone ARKit keeps a stable 60Hz refresh rate, the latency is >100ms between the camera movement and the pose update (we used slow-motion camera to measure the latency), which is consistent with the result in [this paper](https://dl.acm.org/doi/pdf/10.1145/3489849.3489878) (in chapter *Discussion and Future Work*). We guess that there could be some synchronization process in the ARKit where it has to wait for other sensors or data pipelines. Due to this problem, our robot will start shaking if the controller is not trained with latency (in the RL environment). We also tried to use IMU with the legged robot Jacobian to estimate the base velocity. It didn't work well as there could be some bugs in our implementation that we haven't discovered yet.
- iPhone camera pose tracking heavily relies on the number and quality of visual features in the camera view. It works more reliable in-the-wild than in a room with empty walls and floors. Having iPhone slightly pointing up can be helpful to get more visual features, especially when the robot is bending down its back legs. Therefore, we use a 60 degree iPhone mount rather than a 90 degree mount. 
- iPhone camera pose tracking drifts heavily during dynamic movements such as tossing. In our tossing evaluations, the robot is much more likely to misstep using iPhone than mocap. More efforts are needed to improve the robustness of iPhone camera tracking.

Throughout our experiments, mocap is still the most reliable pose estimation method with the lowest latency (<10ms) but cannot work in-the-wild. We are actively searching for better on-board pose estimation devices. Please let us know if you have better solutions!

## GoPro

GoPro with a fish-eye lense is a great solution to simultaneously record videos and camera pose tracking: it is compact, lightweight and easy to use. That said, after extensively using this system, we believe there is still much room for improvement:

- If the scenario for data collection is not bright enough and does not have enough visual features, the open-sourced ORB-SLAM3 fails to track camera poses very frequently. This happens especially when we are facing a blank wall or when the gripper is pointing to the ground. Running more mapping rounds and run batch SLAM multiple times can be helpful. We believe this problem can be solved with more advanced SLAM algorithms or off-the-shelf pose tracking hardware.
- When the camera view is blocked by large objects (e.g. pushing a box or opening a door), the SLAM algorithm may think the camera is not moving. A better solution would be having more tracking cameras facing sideways, which is less likely to be occluded and can still track the gripper movement relative to the environment.
- It is hard for beginners (who is not familiar with this system yet) to collect high quality data for robots to replay. This is due to the kinematic and dynamic infeasibility of robot hardware and controllers. It would be nice if the device can provide instructions and feedback during the data collection time, instead of after running the entire SLAM pipeline and deploying it on the robot.
- When deploying the trained policy to the real-world robot, GoPro image output has to go through a capture card before connecting to PC. This usually introduces ~100ms latency and may not be optimal for reactive tasks.

In general, we are very positive to such lightweight data collection devices and we are looking forward to more improvements in the future.

