The challenge is to find the optimal pathway to net zero UK carbon emissions in 2050, through controlling the rate of deployment of three zero-carbon technologies: offshore wind, blue hydrogen (that is, hydrogen produced from natural gas combined with carbon capture), and green hydrogen (produced from water and renewable electricity by electrolysis). Each run ('episode') of the challenge has 20 time steps representing the years 2031 to 2050. At step `t = 0,1,…,19` you choose the deployment (additional amount of each technology to be built) during the year `2031 + t`, and receive this reward for the year: 

```
 Revenue – (emissions cost + capital cost + operating cost) + t * (jobs created)
```

Clearly, accelerating the deployment of technologies towards net zero reduces total carbon emissions and creates jobs. However it also tends to increase costs, since technology costs tend to reduce over time. Your goal is to find the best balance by maximising the sum of all rewards between year 2031 and 2050.

#### Economic model

The economic model is adapted from the Gale scenario in the [Integrated Energy Vision](https://ore.catapult.org.uk/press-releases/reimagining-a-net-zero-north-sea-an-integrated-energy-vision-for-2050/) study, a collaboration between the Net Zero Technology Centre and Offshore Renewable Energy Catapult. It has these implicit features:

* Sufficient natural gas is imported each year to ensure that demand for both electricity and energy is met;

* Sufficient carbon capture, utilisation and storage (CCUS) technology is deployed each year to ensure that a predetermined, increasing proportion of carbon emissions is captured each year. This proportion is 100% in 2050.

#### Present and future costs

Capital costs, operating costs, emissions costs and revenues in future years are uncertain. Nevertheless, present costs carry information about future costs: lower costs in year `t` typically imply lower costs in future years `t+1, t+2, …` ; likewise, higher present costs typically imply higher future costs. The costs are randomly generated, so will be different for each episode (unless you deliberately fix the random seed).

The challenge has two modes: `open-loop` and `closed-loop`:

* In `open-loop` you observe only the value of the time step `t`

* In `closed-loop` you observe the value of the time step `t` and all present costs/revenues.

#### Actions

Each episode begins in 2030, with all technologies deployed according to the central IEV scenario (Gale) for that year.

At step `t` your agent must choose the additional amount of each technology to deploy in the year `2031 + t`:

* Offshore wind power
* Blue hydrogen
* Green hydrogen

These deployments have upper limits.

#### Scoring

Your score in the competition will be equal to the total reward received by your agent across 1000 episodes. Each entry will be evaluated using the same set of 1000 random seeds.

#### Support

All registered competitors should receive an invitation to the RangL Slack workspace where you can network, get support, and more.

#### Computing requirements

To start developing your agent you will need to install the following: 

* git

* pip 

* Python 3

To submit your agent you will also need to install [Docker](https://www.docker.com/).

#### Code

The code needed for this challenge is available as a Git repository at https://github.com/rangl-labs/netzerotc.

To obtain it, install Git and run

```
git clone https://github.com/rangl-labs/netzerotc.git
```

and head to the README files.

RangL uses the [Openai Gym](https://gym.openai.com/) framework and the challenge is an Openai Gym environment called `env`.

For example, the following call at time `t`:

```python
env.step([3, 0.5, 1]) 
```

corresponds to these additional deployments in that year:

* 3 TWh of offshore wind capacity
* 0.5 TWh of blue hydrogen capacity
* 1 TWh of green hydrogen capacity

This call implements these deployments and returns these objects (see the gym documentation [here](https://gym.openai.com/docs/)): 

* `observation` (the current time step `t`, and (in `closed-loop` mode only:) all present costs/revenues)

* `reward`

* `done` (boolean variable indicating if the time horizon `t=19` has been reached)

You are also provided with helper methods such as 

```python
env.plot()
```

which visualises what has happened in the current episode, and

```
train_rl()
```

which trains a reinforcement learning agent on this challenge.

#### Hint

Modify the environment to train your agent in a smarter way. However, your agent will be evaluated using the original environment.

#### Submission

Submit your entries on or before 31st January 2022 (UK time).



The longer-term goal of the [RangL project](https://www.turing.ac.uk/research/research-projects/ai-control-problems) is to accelerate the use of RL in real-world energy (and other infrastructure) problems by bringing industry, academia and interested users closer together, to understand what kinds of solution work best for different kinds of problem. If you would like to help shape the future development of the platform and its challenges, while gaining experience of agile software development, you are welcome to join our open-source project by contacting [rangl@turing.ac.uk](mailto:rangl@turing.ac.uk).

© 2022, The RangL Team
