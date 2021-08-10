# RangL Pathways to Net Zero challenge

Welcome to the RangL Pathways to Net Zero challenge repository! 

_**Disclaimer:** This repository is a work in progress, as we develop the challenge details. Any file unmodified since 21st June 2021 may not  work (yet)._

To get started, read the challenge overview.

## Challenge overview

Your goal is to find the optimal pathway to net zero carbon emissions for the offshore industry.

Three possible pathways named Breeze, Gale and Storm have been identified in the [Integrated Energy Vision](https://ore.catapult.org.uk/press-releases/reimagining-a-net-zero-north-sea-an-integrated-energy-vision-for-2050/) study by the Net Zero Technology Centre and the Offshore Renewables Catapult. 

In this challenge you can explore the following variations on Breeze, Gale and Storm:

* Implement a weighted combination of the three pathways, and
* Control the rate at which each strategy is implemented.

At the first timestep you choose the scenario weights (they do not have to sum to 1, but they should not be negative). At each timestep you can do this to Breeze, Gale or Storm:

* Pause, by advancing 0 years
* Progress, by advancing 1 year
* Accelerate, by advancing more than 1 year.

Clearly, accelerating progress towards net zero reduces total carbon emissions. However it also tends to be more expensive, since technology costs tend to reduce over time. Your goal is to find the best balance.

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
