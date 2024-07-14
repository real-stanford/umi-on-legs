# Manipulation-centric Whole-body Controller

## Train

To launch WBC training, you'll need to at least specify which device to train and simulate on, as well as the path to the trajectory pickle file
```sh
python scripts/train.py env.sim_device=cuda:0 env.graphics_device_id=0 env.tasks.reaching.sequence_sampler.file_path=wbc/data/tossing.pkl
```

If you are familiar with [Hydra](https://hydra.cc/), the config library, then you'll quickly be able to navigate the `config/` directory and find all hyperparameters you need.
However, incase you aren't, here are relevant hyperparameters for tuning a typical whole-body controller and how to set them.
For each of these hyperparameters, you can just append them to the command. 
For instance, to override how many learning iterations training goes on for
```sh
python scripts/train.py env.sim_device=cuda:0 env.graphics_device_id=0 env.tasks.reaching.sequence_sampler.file_path=wbc/data/tossing.pkl runner.max_iterations=42
```

[](#robustifying-sim2real)

### Training Hyperparameters

- `runner.max_iterations`: sets how many RL learning iterations training goes for.
- `runner.num_transitions_per_env`: sets how many environment steps to rollout each environment for within one on-policy batch. I've found that setting this to be much higher than [`legged_gym`](https://github.com/leggedrobotics/legged_gym)'s default of 24 when you're finetuning the controller for safety and real world deployment leads to much much higher survival rates (see [Robustifying Sim2Real](#robustifying-sim2real)). For instance, I've seen big gains up to 128.
- `runner.alg.num_learning_epochs`: sets how many epochs each on-policy batch of data is trained for. Increasing this didn't help much when training locomotion policies, but it did speed up training significantly when training WBCs. The default value of `32` is much higher than `legged_gym`'s default of `5`.
- `runner.alg.min_lr`: sets the minimum learning rate permitted when using the adaptive learning rate scheduler, based on policy entropy. For this task, lower policy entropy leads to more precise reaching policies, which quickly causes the KL between pre-update and post-update policies to be quite large. This can quickly stall training if `runner.alg.min_lr` is set too low. I've chosen the highest value that gives stable policy optimization in my experience.
- `ckpt_path`: sets the checkpoint path to load in, for either evaluation or for finetuning.

### Environment Set Up & Hyperparameters

- `env`: this sets the base environment configuration, which is where I define all combinations of robot setups, tasks, and training distribution. Ones I've included are the tasks `locomotion`, `locomotion6d`, and `reaching` for `go2` (except for reaching) as well as `go2ARX5`. `locomotion` only has planar base velocity commands, `locomotion6d` extends `locomotion` with height, pitch, and roll commands, while `reaching` is the task proposed in UMI on Legs. For instance, if you want to train a locomotion controller for Go2, you can set `env=combo_go2_locomotion`.
- `env.headless`: sets whether or not a GUI window of the simulation will be created. 
- `env.attach_camera`: is a boolean flag, which, when set to false, will disable rendering and allow training to be ran on a headless server that doesn't have fake displays set up yet. When `env.attach_camera=false`, make sure to set `env.graphics_device_id=-1` as well.
- `env.cfg.terrain.mode`: sets whether the robot will be trained on a flat `plane` or with a `perlin` random noise terrain. In UMI on Legs, we found that training with `env.cfg.terrain.mode=plane` was sufficient for our use case.
- `env.constraints.action_rate.penalty_weight`: sets how much weight the action rate penalty plays in the total reward. This is the hyperparameter we tune the most to get the behavior we want. You can set the penalty weight of all other constraints listed in the `combo_*` training files. For instance, in our default training config `combo_go2ARX5_pickle_reaching_extreme.yaml`, we inherit constraints for `action_rate`, `joint_acc`, `collision` `joint_limit`, `torque`, `even_mass_distribution`, `feet_under_hips`, `aligned_body_ee`, and `root_height` from `combo_go2ARX5_pickle_reaching.yaml`. You can set the penalty weight of any of those with `env.constraints.{}.penalty_weight` where `{}` is the constraint name.

## Robustifying Sim2Real

### Latency

There are multiple important sources of latencies, which has to be tuned for any application.
Note that latency is a function of the policy inference time (how fast can it run), the deployment code (how efficient is it), the kernel of the computer (is it a real-time kernel or just a normal kernel), etc.
You should tune these sources of latencies for your own system if you choose to deploy only parts of our system:
 - **Motor Execution Latency**: There is always a delay between the moment the command is sent to the controller and the moment that command is physically realized. A safe value is 20ms, and we found this is the right value for both our quadruped and our arm. You can override `env.ctrl_delay.data` to train your controller with the right latency.
 - **Pose Estimation Latency**: When doing world-frame/task-frame tracking, the pose estimator will always have some delay. The Optitrack motion capture system we used had an internal system latency of 7ms, on top of whatever network communication overhead. In practice, we found that setting `env.tasks.reaching.pose_latency=0.010` (i.e., 10 milliseconds) gave stable deployment while not being overly dampened (in the case that pose latency is too high). The iPhone technically should have a pose latency of around 140ms based on our measurements. We've tried to implement IMU + proprioceptive velocity estimation to integrate received poses forward in time by 140ms, but we didn't find that this improved the controller's behavior.
 - **Timing**: The controllers were trained to be ran at 50Hz. However, in reality, the controller will never be called exactly 20ms apart with updated state information taken the instant before the controller is called. There is information delay as well as timer inaccuracies. To account for this, we implemented `env.controller.decimation_count_range` which takes a 2-tuple argument, setting the upper and lower bound of how many `dt`s each action will be repeated for. This randomizes the controller callback frequency during training to get the controller more robust to timing inaccuracies.

### Action Rate

Before deploying any controller to real, I would launch a few short finetune training, sweeping action rate penalty weights.
I would then start by deploying the controller with the highest action rate penalty weight that still does the task, for safety, and slowly make my way up to more performant but potentially more dangerous controllers with lower action rate penalty.
These runs usually have `runner.num_transitions_per_env=128`, `runner.alg.num_learning_epochs=64`, and I'd usually get a reasonable checkpoint before learning iteration 500.



## Extending

### More Robots

Looking at `env_go2.yaml` and `env_go2ARX5.yaml` will give you examples for what needs to be defined for each new robot.
I recommend keeping robot specific configs and training (i.e., robot + task combination) configs separate, and have the latter inherit from the former.
This allows you to quickly try out different combinations of robots and tasks.

In my opinion, the most important thing to figure out for any new embodiment is the `env.controller.scale`, `env.controller.kp`, and `env.controller.kd`.
Expect to take a few iterations before you find some good values, and make sure to go back and forth between sim and real before settling on your values.
Here are some tips for tuning these parameters:
 - Is the robot just jumping around crazily in the beginning? If so, use a smaller controller scale. Is the robot's range of movement at the beginning of training too small? If so, consider increasing the controller scale.
 - Is it way too stiff? Could it be more compliant? If so, try decreasing KP. Instead, does it look like the robot is trying a lot but none of its actions are able to move its body? In this case, increase KP, and make sure that the robot's torque limit is not being hit (a dirty little print statement inside `PDController` should do the trick).

While tuning these parameters, you can disable all domain randomization. Assuming that your robot's configuration under no domain randomization is at the center of your domain randomization range (e.g., robot's default mass is 5kg and you randomize between 4kg and 6kg), then the same controller scale, Kp, and Kd should still work once you add domain randomization back in.

### More Manipulation Trajectories

The format expected for WBC training is a pickled list of dictionaries.
Each dictionary contains the pose information for one episode, and must include a `(T, 3)` numpy array under the key `ee_pos` and a `(T, 3)` numpy array under the key `ee_axis_angle`, representing the end effector's position and orientation as axis angles, respectively.
The codebase assumes that these sequences are sampled at 200Hz.

If you're collecting more trajectories using UMI, the you can use `scripts/preprocess_umi_trajs.py`, which will automatically preprocess the trajectories for you from the `dataset_plan.pkl` pickle file (present in every UMI dataset preprocessing result).  

If you're hoping to procedurally generate your own trajectories, you can refer to `scripts/preprocess_umi_trajs.py`, `scripts/generate_pushing_trajectories.py`, and `scripts/generate_random_trajectories.py`.