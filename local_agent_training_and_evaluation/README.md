# Getting started

To train a reinforcement learning agent, first install the Python requirements:

```
pip install -r requirements.txt
```

Any reinforcement learning library can be used. Here, we use Stable Baselines 3. The `util.py` helper contains the `Trainer` class. Its `train_rl` method is an easy way to

- Train an RL agent (using just one episode, so it probably won't win!)
- Save it as `./saved_models/MODEL_0.zip` (see https://stable-baselines3.readthedocs.io/ for more information)

This is illustrated by the `./train.py` script.

The trained models' performance can be evaluated by using several evaluation scripts, such as `evaluate_standard_agents.py` and `evaluate.py`, to calculate the mean rewards and visualize the agents' actions. 

You can also force using a specific random seed to evaluate the agent's performance, for example, in `evaluate_random_agent.py`, the random agent is evaluated using a single `seed=123456`. However, notice that this competition will use a list of random seeds saved in `seeds.csv` file and rank the performance by mean rewards averaged among those random seeds. 

When you have trained and saved multiple models, you can use `plot_learning_curve.py` to plot the learning curve of those trained models, which should tell you how well your trained models are learning from the environment.