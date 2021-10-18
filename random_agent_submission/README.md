# Agent testing and submission

This folder contains 
* `agent.py`, a simple agent which performs random actions for testing purposes
* `test_container.py`, a script which tests locally that `agent.py` works in a container
* the files necessary to submit an agent to the competition platform

Install the dependencies:

```shell
pip install -r requirements.txt
```

Build a docker image for the submission:

```shell
make build
```

At present this will create a local image named `submission:v0.1.0`.

## Local testing

We have included a testing script allowing the agent to be tested with:

```shell
python test_container.py
```

For this to work, the following must hold:

1. An environment container named `nztc` must be running on the docker network named `evalai_rangl`. To create the appropriate container, run the following:

   ```shell
   cd environment
   docker-compose up --build
   ```

Typical output is as follows:

```shell
$ ./test_container.py
DEBUG:__main__:Created submission
DEBUG:__main__:Completed submission
output
51458b8b -55459.6455078125
...
51458b8b 37825.09130859375
51458b8b
done True
51458b8b
DEBUG:__main__:Instance id: 51458b8b
score: 511894.40615844727
...
Evaluation completed using 5 seeds.
Final average score:  492320.25009155273
```

## Submission

The image can be submitted as follows:

1. Install EvalAI

   ```shell
   pip install evalai
   ```

2. Set the EvalAI host

   ```
   evalai host -sh http:submissions.rangl.org
   ```

3. Login

   ```shell
   evalai login
   ```

4. Submit the agent:

   ```shell
   evalai push submission:v0.1.0 --phase nztc-dev-5
   ```



