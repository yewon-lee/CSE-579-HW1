# Homework-1

## Setup and Installation

### Install MuJoCo

1. Download the MuJoCo version 2.1 binaries for
   [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
   [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
3. Add resources/mjkey.txt in the repo into into `~/.mujoco/mujoco210`.

### Setup environment

To set up the project environment, Use the `environment.yml` file. It contains the necessary dependencies and installation instructions.

    conda env create -f environment.yml
    conda activate cse579a1

### Install LibGLEW

    sudo apt-get install libglew-dev
    sudo apt-get install patchelf
    
### Export paths variables
For these, put them in your ~/.bashrc

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    
### Compile mujoco_py (only needs to be done once)
    python -c "import mujoco_py"

## Running the assignment
This is the command you use to run  the assignment:

    python main.py --env reacher/pointmaze --train behavior_cloning/dagger/diffusion --policy gaussian/autoregressive/diffusion

For example if you wanted to run reacher with behavior cloning and gaussian policy you would run:

    python main.py --env reacher --train behavior_cloning --policy gaussian

## Files you need to touch:
More details in the assignment spec TODO PUT LINK.
- main.py (only for hyper parameter tuning) 
    - The assignment will ask you to change certain hyperparameters in main.py like the batch_size or number of training steps.
- DiffusionPolicy.py (for the extra credit)
    - There are three different TODO blocks to implement diffusion policies.
- dagger.py (for your implementation of dagger)
    - There are two different TODO blocks to implement dagger.
- bc.py (for your implementation of bc)
    - There are one different TODO blocks to implement bc.
- utils.py (for the autoregressive model)
    - There is two different TODO blocks to implement the autoregressive model.
