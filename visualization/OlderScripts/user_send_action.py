#%% 

#%% 
import logging

import pandas as pd
import numpy as np
import gym
import argparse
import random
import time

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from pythonosc.dispatcher import Dispatcher
from typing import List, Any

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

dispatcher = Dispatcher()

done = False
next_step = 0
agent_action = np.zeros(3)
def set_agent_action(address: str, *args: List[Any]) -> None:
    global done
    global agent_action
    global next_step

    # Check that address starts with agent_action
    if not address[:-1] == "/agent_action":  # Cut off the last character
        return

    # value1 = args[0]
    # value2 = args[1]
    # value3 = args[2]
    # user_action_no = address[-1]
    # print(f"Setting agent_action {user_action_no} values: {value1}, {value2}, {value3}")
    set_agent_action.done = args[0]
    set_agent_action.values = np.array([args[1], args[2], args[3]])
    done = args[0]
    agent_action = np.array([args[1], args[2], args[3]])
    next_step = args[4]

dispatcher.map("/agent_action*", set_agent_action)  # Map wildcard address to set_agent_action function

# Specify the OSC client and server address
client = SimpleUDPClient("127.0.0.1", 1338)
server = BlockingOSCUDPServer(("127.0.0.1", 1337), dispatcher)

user_action = agent_action
while not done:
    server.handle_request()
    if done:
        break
    print("\n\nThe current env.state.step_count is %d; The Storm/RL agent's action of the next step %d for increments in [offshore Wind, blue Hydrogen, green Hydrogen] are: [%f, %f, %f]" %(next_step-1, next_step, agent_action[0], agent_action[1], agent_action[2]))
    steps_back = abs(np.float64(input("To rewind the state, enter the decrement to step back (any non-zero, + or - integer); Otherwise, press '0' to edit or accept this agent's action and step forward in time: ")))
    steps_back = np.minimum(next_step, steps_back) # clip the # of steps back such that env.state.step_count can be rewound back to -1 at most, equivalent to env.reset() and then without any env.step()
    if not steps_back:
        accept_agent_action = input("Do you want do accept this agent's action? ('n' to input manually / 'y' or any other key to accept) ")
        if (accept_agent_action.lower() == 'n'):
            user_action[0] = np.float64(input("Enter the increment in offshore Wind: "))
            user_action[1] = np.float64(input("Enter the increment in blue Hydrogen: "))
            user_action[2] = np.float64(input("Enter the increment in green Hydrogen: "))
        else:
            user_action = agent_action
    else:
        user_action = agent_action
    client.send_message("/user_action1", np.hstack((steps_back, user_action)))
