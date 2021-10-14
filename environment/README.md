# RangL reference environment

## Installation

To install the environment, first copy a python object serialised from a suitable economic model spreadsheet with the filename `PathwaysToNetZero_Simplified_Compiled.pkl` into the `compiled_workbook_objects` folder.

## Overview

The `reference_environment` folder contains the environment used in the competition. To modify it for development purposes, check out its `README`.

Modified environments can be checked using the `test_reference_environment_direct_deployment.py` script. This will
* Run logical tests, which
    * validate that the environment is correctly specified
    * illustrate useful concepts such as random seeding
* Run a rule-based agent and create a plot `fixed_policy.png` of the episode
 


