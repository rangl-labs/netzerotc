# RangL Pathways to Net Zero challenge

Welcome to the RangL Pathways to Net Zero challenge repository! 

To get started, read the challenge overview.

## Challenge overview

Your goal is to find the optimal pathway to net zero carbon emissions for the offshore industry, by controlling the rate of deployment of three specific technologies between 2031 and 2050. The technologies are offshore wind, blue hydrogen (that is, hydrogen produced from natural gas), and green hydrogen (produced from water and renewable electricity by electrolysis).

The economic and energy system model is adapted from the [Integrated Energy Vision](https://ore.catapult.org.uk/press-releases/reimagining-a-net-zero-north-sea-an-integrated-energy-vision-for-2050/) study by the Net Zero Technology Centre and the Offshore Renewable Energy Catapult. The IEV deploys carbon capture, utilisation and storage (CCUS) as required to ensure net zero by 2050, depending on the chosen deployment of other technologies. The technology mix in 2030 is based on the IEV 'Gale' scenario.

Each timestep corresponds to one year. Beginning in 2031, at each timestep you can choose an amount of each technology to deploy in the coming year. For example, timestep `t` corresponds to calendar year `2030 + t` and the action (3, 0.5, 1) corresponds to these additional deployments in that year:

* 3 GW of offshore wind capacity
* 0.5 TWh of blue hydrogen capacity
* 1 TWh of green hydrogen capacity

The reward at each timestep is given by the formula

`Revenue - (Capex + Opex + Decomm + Emissions)`

whose components are respectively revenue, capital expenditure, operating expenditure, decommissioning costs, and emissions costs.

Clearly, accelerating the deployment of technologies towards net zero reduces total carbon emissions. However it also tends to increase costs, since technology costs tend to reduce over time. Your goal is to find the best balance.

## The RangL environment

RangL uses the [Openai Gym framework](https://gym.openai.com). To install the RangL environment on your local machine, 

1. If necessary, install the pip package manager (you can do this by running the `get-pip.py` Python script)

2. Run `pip install -e environment`

Then head into the `environment` folder and check out the README there.

## Developing your agent

Write Python code which returns your desired action at each timestep. The performance of this code (your _agent_) will determine your score in the challenge. 

Design freely: you might use explicit rules, reinforcement learning, or some combination of these. The helper class `Evaluate` in `local_agent_training_and_evaluation/util.py` illustrates some basic approaches:

* the `min_agent` method gives your actions the smallest possible value at each step
* `max_agent`: largest possible actions at each step
* `random_agent`: random actions at each step
* `RL_agent`: actions are provided by a previously trained RL agent

## Self-evaluation

At each step, the RangL environment uses random noise to model real-world uncertainty. To evaluate an agent yourself, simply average its performance over multiple random seeds. To do this:

* Add your agent to the `Evaluate` helper class (the four agents above are examples)
* Evaluate it just as in `local_agent_training_and_evaluation/evaluate.py`

## Training 

To get started with training RL agents, head to the `local_agent_training_and_evaluation` folder and check out the README.
