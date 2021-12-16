# Getting started

To train a reinforcement learning agent, first install the Python requirements:

```
pip install -r requirements.txt
```

Any reinforcement learning library can be used. Here, we use Stable Baselines 3. The `util.py` helper contains the `Trainer` class. Its `train_rl` method is an easy way to

- Train an RL agent
- Save the trained model (and some intermediate models) as `./saved_models/MODEL_x.zip` (see https://stable-baselines3.readthedocs.io/ for more information)

This is illustrated by the `./train.py` script.
