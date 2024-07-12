<h1> UMI on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers</h1>
<div style="text-align: center;">

[Huy Ha](https://www.cs.columbia.edu/~huy/)$^{🐶,1,2}$, [Yihuai Gao](https://yihuai-gao.github.io/)$^{🐶,1}$ [Zipeng Fu](https://zipengfu.github.io/)$^1$, [Jie Tan](https://www.jie-tan.net/)$^{3}$  [Shuran Song](https://shurans.github.io/)$^{1,2}$

$^1$ Stanford University, $^2$ Columbia University, $^3$ Google DeepMind, $^🐶$ Equal Contribution

[Project Page](https://umi-on-legs.github.io/) | [Arxiv](https://arxiv.org/abs/2307.14535) | [Video](https://www.cs.columbia.edu/~huy/scalingup/static/videos/scalingup.mp4)

<div style="margin:50px; text-align: justify;">
<img style="width:100%;" src="docs/assets/umi_on_legs_toss.gif">


UMI on Legs is a framework for combining real-world human demonstrations with simulation trained whole-body controllers, providing a scalable approach for manipulation skills on robot dogs with arms.


<b>The best part?</b> You can plug-and-play your existing visuomotor policies onto a quadruped, making your manipulation policies mobile!


</div>
</div>

<br>

This repository includes source code for whole-body controller simulation training, whole-body controller real-world deployment, iPhone odometry iOS application, UMI real-world environment class, and ARX5 SDK.
We've published our code in a similar fashion to how we've developed it - as separate submodules - with the hope that the community can easily take any component they find useful out and plug it into their own system.

If you find this codebase useful, consider citing:

```bibtex
@inproceedings{ha2024umionlegs,
      title={{UMI} on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers}, 
      author={Huy Ha and Yihuai Gao and Zipeng Fu and Jie Tan and Shuran Song},
      year={2024},
}
```

If you have any questions, please contact [me](https://www.cs.columbia.edu/~huy/) at `huyha [at] stanford [dot] edu` or [Yihuai](https://yihuai-gao.github.io/) at `yihuai [at] stanford [dot] edu`.

**Table of Contents**

If you just want to start running some commands while skimming the paper, you should [get started here](docs/starter.md), which downloads data, checkpoints, and rolls out the WBC.
The rest of the documentation is focused on setting up real world deployment.
 
 - 🏃‍♀️ [Getting Started](docs/starter.md)
   - ⚙️ [Setup](docs/starter.md#setup)
   - 📍 [Checkpoint & Data](docs/starter.md#downloads)
   - 🕹️ [Rollout](docs/starter.md#rollout-controller)
   - 📊 [Evaluation](docs/starter.md#evaluation)
 - 🦾 [Universal Manipulation Interface]([docs/umi/index.md](https://github.com/real-stanford/universal_manipulation_interface))
   - 📷 [Data Collection](https://swanky-sphere-ad1.notion.site/UMI-Data-Collection-Tutorial-4db1a1f0f2aa4a2e84d9742720428b4c?pvs=4)
   - 🛠️ [Hardware Guide](https://docs.google.com/document/d/1TPYwV9sNVPAi0ZlAupDMkXZ4CA1hsZx7YDMSmcEy6EU/edit)
   - 🎛️ [Preprocessing]([docs/umi/data.md](https://github.com/real-stanford/universal_manipulation_interface?tab=readme-ov-file#running-umi-slam-pipeline))
 - ⚙️ [Manipulation-Centric Whole-body Controller](docs/wbc.md)
   - 🚂 [Train](docs/wbc.md#train)
   - 🛡️ [Robustifying Sim2Real](docs/wbc.md#robustifying-sim2real)
   - 🔭 [Extending](docs/wbc.md#extending)
      - 🤖 [More Robots](docs/wbc.md#more-robots)
      - 🫳 [More Manipulation Trajectories](docs/wbc.md#more-manipulation-trajectories)
 - 🌍 [Real World Deployment](docs/)
   - 📱 [iPhone Odometry]()
   - 🐕 [Quadruped Setup & Deployment]()
   - 🏠 [Deployment Environment]()
 - 📽️ [Visualizations](docs/visualization.md)
 

# Code Acknowledgements

**Whole-body Controller Simulation Training**: 
 - Like many other RL for control works nowadays, we started with [Nikita Rudin](https://scholar.google.com/citations?user=1kKJYVIAAAAJ&hl=fr)'s implementation of PPO and Gym environment wrapper around IsaacGym, [legged gym](https://github.com/leggedrobotics/legged_gym). Shout out to Nikita for publishing such a hackable codebase - it's truly an amazing contribution to our community.
 - Although not used in the final results of the paper, our codebase does include a modified Perlin Noise Terrain from [DeepWBC](https://manipulation-locomotion.github.io/). To use it, run training with `env.cfg.terrain.mode=perlin`.

**Whole-body Controller Deployment**: 
 - Thanks to [Qi Wu](https://wooqi57.github.io/) for providing us with an initial deployment script for the whole-body controller!

**iPhone Odometry Application**: 
 - Thanks to [Zhenjia Xu](https://www.zhenjiaxu.com/) for providing us with some starter code for ARKit camera pose publishing!

**UMI Environment Class**:
 - Our UMI deployment codebase heavily builds upon the original [UMI codebase](https://github.com/real-stanford/universal_manipulation_interface). Big thanks to the [UMI team](https://umi-gripper.github.io/)!