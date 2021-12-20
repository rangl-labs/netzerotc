# Getting started

To train a reinforcement learning agent, first install the Python requirements:

```
pip install -r requirements.txt
```

Any reinforcement learning library can be used. Here, we use Stable Baselines 3. The `util.py` helper contains the `Trainer` class. Its `train_rl` method is an easy way to

- Train an RL agent
- Save the trained model (and some intermediate models) as `./saved_models/MODEL_x.zip` where `x=0,1,2,...` (see https://stable-baselines3.readthedocs.io/ for more information)

This is illustrated by the `./train.py` script.

To evaluate the average performance of trained models across multiple episodes, and to plot their actions in a single episode, this folder contains:

- `evaluate_random_agent.py` -- evaluate the random agent on a fixed seed (`seed=123456`)
- `evaluate.py` -- evaluate the random agent, benchmark agents, and a trained RL agent, with multiple seeds, generate plots, and save in the `saved_models` folder

To see how well the models learn from the environment, train and save multiple models and use the `plot_learning_curve.py` script.
