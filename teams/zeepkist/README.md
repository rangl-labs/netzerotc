# rangl-zeepkist
[![DOI](https://zenodo.org/badge/329377250.svg)](https://zenodo.org/badge/latestdoi/329377250)

## Description
This repository contains the code for two *"zeepkist"* agents submitted to the [RangL Generation Scheduling Challange (Jan 2021)](http://challenge1-rangl.uksouth.cloudapp.azure.com:8888/web/challenges/challenge-page/1/overview). The challenge consists of dispatching two electrical generators to track a given net energy demand over a period of two days, using half-hourly updated forecasts.

The submitted agents are:
* ``zeepkist_rl``: *1st place*, offline runs (averaged over 100 episodes with provided seeds)
* ``zeepkist_mpc``: *1st place*, online leaderboard (single seed)

The *zeepkist team* consists of:
- Simon Tindemans, Delft University of Technology (code)
- Jan Viebahn, TenneT TSO B.V. (support)

## Installation and usage
* Clone repository: ``git clone https://github.com/simontindemans/rangl-zeepkist.git``
* *(Optional, but recommended)* Set up a new python 3.8 environment. Instructions to create a new 'zeepkist' environment using ``conda`` are:
    - ``conda env create -f conda_env.yml``
    - ``conda activate zeepkist``
* If you did *not* install the conda environment using ``conda_env.yml``, then manually install required packages: ``pip install -r rangl-zeepkist/requirements.txt``
* *(Optional)* Run ``python code/train-rl.py`` to train the RL agent (note: this will overwrite ``models/MODEL_0.zip``)
* Run one or more of the three evaluation scripts to evaluate the performance of the agents
    - MPC agent: ``python code/evaluate-mpc.py`` 
    - RL agent: ``python code/evaluate-rl.py``
    - clairvoyant agent: ``python code/evaluate-clairvoyant.py``

## Agent description
### RL agent
The RL agent (v1.1.1) is an SAC agent trained on a modified observation and action space. The **observation space** was modified to consist of the *current time*, *current dispatch* and a *forecast of the next 25 time steps*; the latter makes it easier to embed time invariance into the agent's actions. When necessary (near the end of the interval), the final forecast value was repeated to achieve a 25 time step forecast. The **action space** was modified so that the range *\[-1,1\]* for each generator corresponds to its full ramp range (*-1* is maximum ramp down, *0* is constant generation, *1* is maximum ramp up). The submitted agent was trained on 5000 episodes; evaluation was done deterministically. A relatively small discount rate (gamma) of 0.85 is used to reflect the system's ability to implement arbitrary changes in generation levels within 13 time steps. The trained RL agent is available in ``models/MODEL_0.zip``.

### MPC agent
The MPC agent (v1.1.0) minimises, at each time step, the system cost across the next 25 time steps, based on the most recent demand forecast. It implements the actions for the current time step only, and repeats the procedure when a new forecast is available. This agent uses an explicit representation of the model constraints to construct an MILP problem that is solved using ``pulp``.

### Clairvoyant agent
The clairvoyant agent is only included to compute a lower bound to the cost that is possible with perfect information. It first completes the episode using a constant dispatch, and uses the realised demand trace to the minimum dispatch cost in hindsight. As it is a 'cheating' agent, it can only be run locally and was not submitted to the challenge. 

## Submitted agents
Submitted agents are in the ``submissions`` directory. Testing and submitting these agents requires the installation of Docker. Ensure that the local Docker environment makes sufficient memory available (at least 4GB). 

## License
All code is released under the MIT license. Portions of the code are based on the [RangL Generation scheduling challenge January 2021 repository](https://gitlab.com/rangl-public/generation-scheduling-challenge-january-2021), released (c) RangL Team under the MIT license.
