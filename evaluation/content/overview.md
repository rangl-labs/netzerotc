The challenge is to find the optimal pathway to net zero UK carbon emissions in 2050, through controlling the rate of deployment of three zero-carbon technologies: offshore wind, blue hydrogen (that is, hydrogen produced from natural gas combined with carbon capture), and green hydrogen (produced from water and renewable electricity by electrolysis). Each run ('episode') of the challenge has 20 time steps representing the years 2031 to 2050. At step `t = 0,1,…,19` you choose the deployment (additional amount of each technology to be built) during the year `2031 + t`, and receive this reward for the year: 

```
Revenue – (emissions cost + capital cost + operating cost)
```

Clearly, accelerating the deployment of technologies towards net zero reduces total carbon emissions. However it also tends to increase costs, since technology costs tend to reduce over time. Your goal is to is to find the best balance by maximising the sum of all rewards between year 2031 and 2050, while ensuring that the total number of jobs in the associated industries is maintained at an acceptable level. 

#### Economic model

The economic model is adapted from the Gale scenario in the [Integrated Energy Vision](https://ore.catapult.org.uk/press-releases/reimagining-a-net-zero-north-sea-an-integrated-energy-vision-for-2050/) study, a collaboration between the Net Zero Technology Centre and Offshore Renewable Energy Catapult. It has these implicit features:

* Sufficient natural gas is imported each year to ensure that demand for both electricity and energy is met;

* Sufficient carbon capture, utilisation and storage (CCUS) technology is deployed each year to ensure that a predetermined, increasing proportion of carbon emissions is captured each year. This proportion is 100% in 2050.

#### Present and future costs

Capital costs, operating costs, emissions costs and revenues in future years are uncertain. Nevertheless, present costs carry information about future costs: lower costs in year `t` typically imply lower costs in future years `t+1, t+2, …` ; likewise, higher present costs typically imply higher future costs. You observe all present costs at each time step. The costs are randomly generated, so will be different for each episode (unless you deliberately fix the random seed).

#### Actions

 At each time `t`, your agent must choose the additional amount of each of these three technologies to deploy in that year: 

* Offshore wind
* Blue hydrogen
* Green hydrogen

#### Constraints

There are the following constraints: 

* There are upper limits on how much of each technology can be deployed per year

* The total number of jobs must not decrease too quickly, and must not fall below a given level

If the action violates these constraints, a large penalty is subtracted from that year’s reward.

At the beginning of each episode, the amount of each technology is that given by the IEV Gale scenario.

#### Scoring

Your score in the competition will be equal to the total reward received by your agent across 1000 episodes. Each entry will be evaluated using the same set of 1000 random seeds.

#### Support

All registered competitors should receive an invitation to an 'Ask away!' (`#askaway`) channel on Slack, where you can get support.

#### Computing requirements

To start developing your agent you will need to install the following (in case of difficulty please `#askaway`): 

* git

* pip 

To submit your agent you will also need to install [Docker](https://www.docker.com/).

#### Code

The code needed for this challenge is available as a Git repository at https://github.com/rangl-labs/netzerotc.

To obtain it, install Git and run

```
git clone https://github.com/rangl-labs/netzerotc.git
```

and head to the README files.

RangL uses the [Openai Gym](https://gym.openai.com/) framework. So for example, the following call at time `t`:

```python
env.step([3, 0.5, 1]) 
```

corresponds to these additional deployments in that year:

* 3 GW of offshore wind capacity
* 0.5 TWh of blue hydrogen capacity
* 1 TWh of green hydrogen capacity

This call implements these deployments and returns these objects (see the gym documentation [here](https://gym.openai.com/docs/)): 

* `observation` (the current time step `t`, and all present costs/revenues)

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

Modify the generation scheduling environment to train your agent in a smarter way. However, your agent will be evaluated using the original generation scheduling environment.

#### Submission

Submit your entries on or before 31st January 2022 (UK time).



The longer-term goal of the [RangL project](https://www.turing.ac.uk/research/research-projects/ai-control-problems) is to accelerate the use of RL in real-world energy (and other infrastructure) problems by bringing industry, academia and interested users closer together, to understand what kinds of solution work best for different kinds of problem. If you would like to help shape the future development of the platform and its challenges, while gaining experience of agile software development, you are welcome to join our open-source project by contacting [rangl@turing.ac.uk](mailto:rangl@turing.ac.uk).

© 2022, The RangL Team
