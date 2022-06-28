# Visualization of env.py in Unreal Engine

## Overview

This folder contains a visualization of the `env.py` in Unreal Engine.

The general idea is to use some objects (e.g., wind turbines and oil platforms) in a virtual world with UK geography & satellite images created in Unreal Engine to represent the outputs of `env.render()`.

The `env.render()` can output some values (typically stored in a NumPy array) to be visualized, e.g., the deployment numbers of the 3 technologies, rewards, CO2 emissions, etc. Visualizing these values can provide the end-user an idea of the current status of the `env`.

To send the NumPy array containing these values to Unreal Engine for visualization, we need some type of middleware to communicate with the Unreal Engine. An example is the [Open Sound Control (OSC)](https://en.wikipedia.org/wiki/Open_Sound_Control), which is a protocol originally developed for MIDI but is capable for sending/receiving other types of data, e.g., arrays of floating numbers.

There is an OSC plug-in developed for Unreal Engine to receive numerical array as OSC messages sent from outside Unreal Engine, and there are several implementations of OSC in Python to send and receive OSC messages containing arrays of numeric values. Several examples can be found at https://pypi.org/search/?q=%22open+sound+control%22&o=

For testing purposes, the `render()` method of `env.py` is unchanged. Instead, in the `osc_UE_UMG_env_3FixedPolicies.py` script in this folder, the `python-osc` package is used to send `step_count`, `reward`, `3-element deployments`, and `CO2 emission amount` of the `env` to Unreal Engine for visualization.

The `UE_Projects` folder contains project files of Unreal Engine, and within these files, there is an OSC server built within Unreal Engine using the OSC plug-in to receive the OSC message sent from `osc_UE_UMG_env_3FixedPolicies.py`, and according to the OSC messages received, it will change the physical properties of some objects (e.g., the number of wind turbines and oil platforms built) to represent the deployment numbers in the OSC messages sent from `osc_UE_UMG_env_3FixedPolicies.py`.

<!---
An end-user can control the `env.step()` by manually specifying an action, or simply accept the Storm or RL agent's proposed action. Moreover, the end-user can also rewind the `env.state` backward in time and restart/re-input some new actions. These are implemented in the `user_send_action.py`. The `env` will wait for end-user's input/action before `env.step(action)` or it will be rewound back to a previous step, which are implemented in `send_receive_osc_env.py`.
-->

An end-user can control the `env.step()` in Unreal Ungine's UMG Widgets by manually specifying the scenario (out of Breeze, Gale, or Storm) and the year to visualize; an example is shown in the following screen recording (wait until the 87.8MB .gif animation is downloaded to web browser, and then click on it to shown the animation): ![screenrecording](https://github.com/rangl-labs/netzerotc/raw/visualization/visualization/VideoDemo.gif "Unreal Engine screen recording"); A higher resolution version is available in the video https://github.com/rangl-labs/netzerotc/raw/visualization/visualization/VideoDemo.mp4

For more information of OSC implementations in Python, read the documentations of [python-osc](https://python-osc.readthedocs.io/en/latest/) and [osc4py3](https://osc4py3.readthedocs.io/en/latest/). The history of development can be found at https://github.com/orgs/rangl-labs/projects/1

## Steps to Test the Visualization

To give this visualization a test:

1. Install Unreal Engine at https://www.unrealengine.com/en-US/download/creators?install=true

<!---
2. Within Unreal Engine, open the `BasicVisualization1` project in `UE_Projects` folder. Click the triangle of drop-down menu next to the "Play" button, and then select "Simulate" (Please see the following screenshot ![screenshot](https://github.com/rangl-labs/netzerotc/raw/visualization/visualization/HowToRunUE.png "Unreal Engine screenshot") showing the buttons).
-->

2. Open a Python console and set current directory to this folder, then run `python osc_UE_UMG_env_3FixedPolicies.py`.

3. Within Unreal Engine, open the `BasicVisualization1` project in `UE_Projects` folder. Follow the screen recording below to run or play, and specify scenario and year to visualize (Note: a higher resolution version is available in the video https://github.com/rangl-labs/netzerotc/raw/visualization/visualization/HowToRunUE.mp4): ![screenrecording](https://github.com/rangl-labs/netzerotc/raw/visualization/visualization/HowToRunUE.gif "Unreal Engine screen recording")

<!---
4. Open **another** Python console and run `python send_receive_osc_env.py`.

5. After a while, the **first** Python console will prompt some message asking an end-user to specify or accept agent's action, and then the Python console of `send_receive_osc_env.py` will take user's input and manipulate the `env` by `env.step()` or rewind `env.state` to a previous step, and then send the array of numeric values to Unreal Engine for visualization.
--->

<!---
Note: Depending on your OS, Unreal Engine will stop the real-time rendering if the Unreal Engine's program window is not in focus or on top of the desktop, e.g., when you input some numbers in the Python console of `user_send_action.py`. Therefore, to see the visualization showing/changing in real-time, you may need to switch to Unreal Engine's program window for each step when you manually input in the Python console of `user_send_action.py`.
-->

There are some nice video tutorials of how to navigate the virtual 3D world inside Unreal Engine, both on [YouTube](https://www.youtube.com/watch?v=j2CKS6G3G2k) and [Epic Games websites](https://www.unrealengine.com/en-US/onlinelearning-courses/your-first-hour-in-unreal-engine-4) (free registration required; some people had uploaded it to [YouTube](https://www.youtube.com/watch?v=jNUaR6y7sE4)). A very nice and comprehensive course can also be found on [YouTube](https://www.youtube.com/watch?v=_a6kcSP8R1Y).
