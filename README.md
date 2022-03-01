# RangL Pathways to Net Zero challenge

Welcome to the RangL Pathways to Net Zero challenge repository! 

The challenge took place from 17th to 31st January 2022. After consideration of both the leaderboard and the provided executive summaries, the selection panel chose three joint winners:

* **Epsilon-greedy** (Delft University of Technology)
* **Lanterne-Rouge-BOKU-AIT** (University of Natural Resources and Life Sciences, Vienna, and Austrian Institute of Technology)
* **VUltures** (Vrije Universiteit Amsterdam). 

Additionally, Team **AIM-Mate** were highly commended for their efforts.

## The RangL environment

RangL uses the [Openai Gym framework](https://gym.openai.com). To install the RangL environment on your local machine, 

1. If necessary, install the pip package manager (you can do this by running the `get-pip.py` Python script)

2. Run `pip install -e .`

Then head into the `rangl/env_open_loop` folder and check out the README there.

## Developing your agent

Write Python code which returns your desired action at each timestep. The performance of this code (your _agent_) will determine your score in the challenge. 

Design freely: you might use explicit rules, reinforcement learning, or some combination of these. The helper class `Evaluate` in `meaningful_agent_training/util.py` illustrates some basic approaches:

* the `min_agent` method gives your actions the smallest possible value at each step
* `max_agent`: largest possible actions at each step
* `random_agent`: random actions at each step
* `RL_agent`: actions are provided by a previously trained RL agent

The `Evaluate` class also provides benchmark agents drawn from the [Integrated Energy Vision](https://ore.catapult.org.uk/press-releases/reimagining-a-net-zero-north-sea-an-integrated-energy-vision-for-2050/) study, which your agent should aim to beat:

* `breeze_agent`: implements actions corresponding to the Emerging scenario in the IEV study. These actions focus on deploying offshore wind
* `gale_agent`: implements the Progressive IEV scenario: Higher offshore wind and a mix of blue and green hydrogen
* `storm_agent`: implements the Transformational IEV scenario: Highest offshore wind, paired with green hydrogen

## Self-evaluation

At each step, the RangL environment uses random noise to model real-world uncertainty. To evaluate an agent yourself, simply average its performance over multiple random seeds. To do this:

* Add your agent to the `Evaluate` helper class (the agents above are examples)
* Evaluate it just as in `meaningful_agent_training/evaluate.py` or `meaningful_agent_training/evaluate_standard_agents.py`

## Training 

To get started with training RL agents, head to the `meaningful_agent_training` folder and check out the README.

## Submission

First test submission of a random agent to the competition by heading to the `random_agent_submission` folder and checking out its README. Then head to the `meaningful_agent_submission` folder to submit your competition entry.

## Note

The `evaluation` folder is used to generate the challenge's web front-end and is not relevant to agent development.
