Pre-trained agent: Closed loop
Environment: Closed loop


Test 1:

```
Evaluation completed using 8 seeds.
Final average score:  83324.07849884033
```

Test 2:


```
Evaluation completed using 8 seeds.
Final average score:  83324.07849884033
```

---

Pre-trained agent: Open loop
Environment: Closed loop

```shell
INFO: evaluating seed 0 of 8
Traceback (most recent call last):
  File "/Users/lrmason/github.com/rangl-labs/netzerotc/meaningful_agent_submission/closed_loop/./test_container.py", line 27, in <module>
    submission = client.containers.run(
  File "/Users/lrmason/miniconda3/envs/dev/lib/python3.10/site-packages/docker/models/containers.py", line 848, in run
    raise ContainerError(
docker.errors.ContainerError: Command 'None' in image 'submission-closed-loop:v0.1.0' returned non-zero exit status 1
```

Interactive debugging gives:

```
root@87fcc9c273a2:/service# python agent.py
Traceback (most recent call last):
  File "agent.py", line 35, in <module>
    action, _ = model.predict(obs, deterministic=True)
  File "/usr/local/lib/python3.8/site-packages/stable_baselines3/common/base_class.py", line 473, in predict
    return self.policy.predict(observation, state, mask, deterministic)
  File "/usr/local/lib/python3.8/site-packages/stable_baselines3/common/policies.py", line 281, in predict
    vectorized_env = is_vectorized_observation(observation, self.observation_space)
  File "/usr/local/lib/python3.8/site-packages/stable_baselines3/common/utils.py", line 226, in is_vectorized_observation
    raise ValueError(
ValueError: Error: Unexpected observation shape (16,) for Box environment, please use (1,) or (n_env, 1) for the observation shape.
```

---


Pre-trained agent: Open loop
Environment: Open loop

Test 1:

```
Evaluation completed using 8 seeds.
Final average score:  -57440.163009643555
```

Test 2:

```
Evaluation completed using 8 seeds.
Final average score:  -57440.163009643555
```


---

Pre-trained agent: Closed loop
Environment: Open loop


```
INFO: evaluating seed 0 of 8
Traceback (most recent call last):
  File "/Users/lrmason/github.com/rangl-labs/netzerotc/meaningful_agent_submission/open_loop/./test_container.py", line 27, in <module>
    submission = client.containers.run(
  File "/Users/lrmason/miniconda3/envs/dev/lib/python3.10/site-packages/docker/models/containers.py", line 848, in run
    raise ContainerError(
docker.errors.ContainerError: Command 'None' in image 'submission-open-loop:v0.1.0' returned non-zero exit status 1
```

Interactive debugging gives:

```
root@de9557b4c7a4:/service# python agent.py
Traceback (most recent call last):
  File "agent.py", line 35, in <module>
    action, _ = model.predict(obs, deterministic=True)
  File "/usr/local/lib/python3.8/site-packages/stable_baselines3/common/base_class.py", line 473, in predict
    return self.policy.predict(observation, state, mask, deterministic)
  File "/usr/local/lib/python3.8/site-packages/stable_baselines3/common/policies.py", line 281, in predict
    vectorized_env = is_vectorized_observation(observation, self.observation_space)
  File "/usr/local/lib/python3.8/site-packages/stable_baselines3/common/utils.py", line 226, in is_vectorized_observation
    raise ValueError(
ValueError: Error: Unexpected observation shape (1,) for Box environment, please use (16,) or (n_env, 16) for the observation shape.
```