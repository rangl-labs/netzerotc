# RangL generation scheduling challenge

Welcome to the RangL generation scheduling challenge repository! To get started, read the [challenge overview](http://challenge1-rangl.uksouth.cloudapp.azure.com:8888/web/challenges/challenge-page/1/overview).

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

## Submitting an agent

**Note:** Agent submission is a Beta feature, please use Slack for feedback and support

To submit an agent: 

1. Edit `meaningful_agent_submission/agent.py` [here](https://gitlab.com/rangl-public/generation-scheduling-challenge-january-2021/-/blob/3b181110795c2a08e6f9045ef25a8f061fa31564/meaningful_agent_submission/agent.py#L33) , so that `action` is the action returned by your agent. Ensure that any supporting files needed by your agent are also in the `meaningful_agent_submission` folder
2. Create an account at the [RangL competition platform](http://challenge1-rangl.uksouth.cloudapp.azure.com:8888) (one per team is sufficient) - do not attempt to verify it by email, instead...
3. Let us know your team name on Slack and we will manually approve it
4. Log in to the platform and go to All Challenges -> View Details of the Generation Scheduling Challenge -> Participate -> under My Participant Teams tick the pre-initialised team -> click Continue -> Accept the T&Cs
5. Follow the [Submission Guidelines](http://challenge1-rangl.uksouth.cloudapp.azure.com:8888/web/challenges/challenge-page/1/submission) given on the platform
 
