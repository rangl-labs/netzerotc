# RangL reference environment

## Overview

The `env_open_loop` and `env_closed_loop` folders contain the environments used in the competition. To modify them for development purposes, look at their `README` files.

Modified environments can be checked using the `test_reference_environment.py` script. This will
* Run logical tests, which
    * validate that the environment is correctly specified
    * illustrate useful concepts such as random seeding
* Run a rule-based agent and create a plot `fixed_policy_DirectDeployment.png` of the episode
 
Modified environments can also be tested using the `test_random_agent.py` script. This will run an agent taking random actions on each step, and then it will calculate its averaged performance in terms of mean reward.

If you modify the environment and change the name of the folder containing `env.py`, you will need to update the `setup.py` script and re-run `pip install -e environment` to register the new folder name (see README.md at https://github.com/rangl-labs/netzerotc). Alternatively, you can also manually import the new folder name before creating the environment by `gym.make()`.

The `compiled_workbook_objects` folder contains the economic model used in the environment and should not be altered.

The `server.py` script is used for remote evaluation and is not needed for local training and development.
