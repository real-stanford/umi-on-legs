# Starter

## Set Up

Create the conda environment. I *highly* recommend using [Mamba](https://mamba.readthedocs.io/en/latest/installation.html), which is [faster](https://blog.hpc.qmul.ac.uk/mamba.html#:~:text=mamba%20is%20a%20re%2Dimplementation,Red%20Hat%2C%20Fedora%20and%20OpenSUSE), but if you insist on Conda, then replace `mamba` commands below with `conda`
```sh
mamba env create -f isaac.yml
mamba activate isaac
```

Then, in the repo's root, install the repo as a pip package
```sh
pip install -e .
```

## Downloads

Download manipulation trajectories preprocessed for the whole-body controller pipeline
```sh
wget -qO- http://real.stanford.edu/umi-on-legs/wbc/data.zip | bsdtar -xvf- -C ./
```
as well as the pretrained checkpoints
```sh
wget -qO- http://real.stanford.edu/umi-on-legs/wbc/checkpoints.zip | bsdtar -xvf- -C ./
```

## Rollout Controller

To visualize the controller in simulation
```sh
python scripts/play.py --ckpt_path wbc/checkpoints/tossing/ours-real/model.pt --trajectory_file_path wbc/data/tossing.pkl --device cuda:0 --num_steps 1000 --num_envs 1  --visualize
```

This will also dump out the states needed for Blender visualization.
See the [visualization instructions](./visualization.md) for more details.


> 🪲 Troubleshooting IsaacGym
>
> A known issue with IsaacGym installation is that library paths aren't correctly updated, leading to the following error message
> ```
> ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
> ```
> To bypass this, add your `mamba` environment's library path to the library path environment variable by prepending commands with 
> `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/mambaforge/envs/isaac/lib/`

## Evaluation

To evaluate our controller in simulation
```sh
python scripts/evaluate.py env.sim_device=cuda:0 env.graphics_device_id=0 env.cfg.env.episode_length_s=17.0 env.tasks.reaching.sequence_sampler.file_path=wbc/data/tossing.pkl ckpt_path=wbc/checkpoints/tossing/ours/model.pt env.tasks.reaching.target_obs_times="[-0.06,-0.04,-0.02,0.0,0.02,0.04,0.06,1.0]" 
```
The summary that gets printed out at the end by weights and biases gives the metrics we care about.
 - `eval/task/reaching/pos_err/mean` and `eval/task/reaching/orn_err/mean` gives the position and orientation errors in meters and radians, respectively.
 - `eval/time_outs/sum` gives the proxy for survival rate. If the robot didn't terminate midway through the episode, then the episode should have timed out.
 - `eval/constraint/energy/sum_electrical_power/mean` gives the average power usage in Watts.

To evaluate the no-preview baseline in simulation
```sh
python scripts/evaluate.py env.sim_device=cuda:0 env.graphics_device_id=0 env.cfg.env.episode_length_s=17.0 env.tasks.reaching.sequence_sampler.file_path=wbc/data/tossing.pkl ckpt_path=wbc/checkpoints/tossing/no-preview/model.pt env.tasks.reaching.target_obs_times="[0.0]"
```

To evaluate the body-space baseline in simulation
```sh
python scripts/evaluate.py env.sim_device=cuda:0 env.graphics_device_id=0 env.cfg.env.episode_length_s=17.0 env.tasks.reaching.sequence_sampler.file_path=wbc/data/tossing.pkl ckpt_path=wbc/checkpoints/tossing/body-space/model.pt env.tasks.reaching.target_obs_times="[-0.06,-0.04,-0.02,0.0,0.02,0.04,0.06,1.0]" env.tasks.reaching.target_relative_to_base=true env.tasks.reaching.pos_obs_scale=1.0
```


To evaluate the random trajectories baseline in simulation
```sh
python scripts/evaluate.py env.sim_device=cuda:0 env.graphics_device_id=0 env.cfg.env.episode_length_s=17.0 env.tasks.reaching.sequence_sampler.file_path=wbc/data/tossing.pkl ckpt_path=wbc/checkpoints/tossing/random-trajs/model.pt env.tasks.reaching.target_obs_times="[-0.06,-0.04,-0.02,0.0,0.02,0.04,0.06,1.0]" 
```

To evaluate the DeepWBC baseline in simulation
```sh
python scripts/evaluate.py env.sim_device=cuda:0 env.graphics_device_id=0 env.cfg.env.episode_length_s=17.0 env.tasks.reaching.sequence_sampler.file_path=wbc/data/tossing.pkl ckpt_path=wbc/checkpoints/tossing/deepwbc/model.pt env.tasks.reaching.target_obs_times="[0.0]" env.tasks.reaching.target_relative_to_base=true env.tasks.reaching.pos_obs_scale=1.0
```