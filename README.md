# RangL Pathways to Net Zero challenge

Welcome to the RangL Pathways to Net Zero challenge repository! 

To get started, read the [challenge overview](http://51.132.63.168:8888/web/challenges/challenge-page/8/overview).

## The RangL environment

RangL uses the [Openai Gym framework](https://gym.openai.com). To install the RangL environment on your local machine, 

1. If necessary, install the pip package manager (you can do this by running the `get-pip.py` Python script as detailed here: https://pip.pypa.io/en/stable/installation/#get-pip-py)

2. Run `pip install -e .`

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
* Evaluate it just as in `local_agent_training_and_evaluation/evaluate.py` or `local_agent_training_and_evaluation/evaluate_standard_agents.py`

## Training 

To get started with training RL agents, head to the `local_agent_training_and_evaluation` folder and check out the README.

## Submission

To submit your agent to the competition, head to the `random_agent_submission` folder and check out its README.
