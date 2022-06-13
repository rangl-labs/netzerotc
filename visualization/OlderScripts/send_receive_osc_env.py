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

import reference_environment_direct_deployment

from pathlib import Path

from pythonosc.dispatcher import Dispatcher
from typing import List, Any

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

dispatcher = Dispatcher()

user_action = np.zeros(3)
steps_back = 0
def set_user_action(address: str, *args: List[Any]) -> None:
    global steps_back
    global user_action
    # We expect three float arguments
    # if not len(args) == 3 or type(args[0]) is not float or type(args[1]) or type(args[2]) is not float:
    #     return

    # Check that address starts with user_action
    if not address[:-1] == "/user_action":  # Cut off the last character
        return

    value1 = args[1]
    value2 = args[2]
    value3 = args[3]
    current_step = env.state.step_count
    # current_step = address[-1]
    # print(f"\n \n Setting user_action values of step {current_step} to: [{value1}, {value2}, {value3}]")
    # print("\n\nCurrent env.state.step_count is %d; Setting user_action values of next step %d to: [%f, %f, %f]" %(current_step, current_step+1, value1, value2, value3))
    set_user_action.steps_back = np.int64(abs(args[0]))
    set_user_action.values = np.array([args[1], args[2], args[3]])
    steps_back = np.int64(abs(args[0]))
    user_action = np.array([args[1], args[2], args[3]])
    if not steps_back:
        print("\n\nSetting env.state to step from env.state.step_count = %d to %d, after setting user_action values of next step %d to: [%f, %f, %f] ..." %(current_step, current_step+1, current_step+1, value1, value2, value3))
    else:
        print("\n\nRewinding env.state from env.state.step_count = %d back to %d ..." %(current_step, current_step-steps_back))
        

dispatcher.map("/user_action*", set_user_action)  # Map wildcard address to set_user_action function


# Specify the OSC (in Unreal Engine) server address
ip = "127.0.0.1"
port = 5005
client_toUEosc = SimpleUDPClient(ip, port)  # Create OSC client
client_toUMGpopup = SimpleUDPClient("127.0.0.1", 1336)

# Specify the OSC client and server address
client = SimpleUDPClient("127.0.0.1", 1337)
server = BlockingOSCUDPServer(("127.0.0.1", 1338), dispatcher)


# Create an environment named env
env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
# set the seed equal to current time in rounded seconds
seed = int(time.time())
env.seed(seed)
# Reset the environment
env.reset()

done = False
rewards = []
deployments = np.array(np.array(pd.read_excel('STORM_Deployments_Modified.xlsx'))[-(env.param.steps_per_episode+1):,1:],dtype=np.float32)
actions = deployments[1:,:] - deployments[:-1,:]
while not done:
    action = actions[env.state.step_count + 1]
    client.send_message("/agent_action1", np.hstack((done, action, env.state.step_count + 1)))
    server.handle_request()
    if steps_back:
        previous_actions = env.state.actions_all[:-abs(steps_back)]
        env.seed(seed)
        env.reset()
        for previous_action in previous_actions:
            observation, reward, done, _ = env.step(previous_action)
    else:
        observation, reward, done, _ = env.step(user_action)
    # observation, reward, done, _ = env.step(user_action)
    print("\nValues sent to OSC server in Unreal Engine for visualization are [step_count, reward, 3-element deployments, CO2 emission amount]: " + str(np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount))))
    client_toUEosc.send_message("/some/address", np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount)))
    client_toUMGpopup.send_message("/some/address", np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount)))
    # client.send_message("/some/address", env.render())
    if done:
        client.send_message("/agent_action1", np.hstack((done, action, env.state.step_count)))


# def storm_agent(self, seeds):
#     rewards = []
#     # deployments = np.array(np.array(pd.read_excel('STORM_Deployments.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
#     deployments = np.array(np.array(pd.read_excel('STORM_Deployments_Modified.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
#     actions = deployments[1:,:] - deployments[:-1,:]
#     for seed in seeds:
#         self.env.seed(seed)
#         self.env.reset()
#         while not self.env.state.is_done():
#             action = actions[self.env.state.step_count + 1]
#             self.env.step(action)
#         rewards.append(sum(self.env.state.rewards_all))
#     return np.mean(rewards)



#%% python-osc test:
# import logging

# import pandas as pd
# import numpy as np
# import gym
# import argparse
# import random
# import time

# # from pythonosc import udp_client

# import reference_environment_direct_deployment

# from pathlib import Path

# from pythonosc.dispatcher import Dispatcher
# from typing import List, Any

# dispatcher = Dispatcher()

# testval = 0.0
# def set_filter(address: str, *args: List[Any]) -> None:
#     global testval
#     # We expect two float arguments
#     # if not len(args) == 2 or type(args[0]) is not float or type(args[1]) is not float:
#     #     return

#     # Check that address starts with filter
#     if not address[:-1] == "/filter":  # Cut off the last character
#         return

#     value1 = args[0]
#     value2 = args[1]
#     filterno = address[-1]
#     print(f"Setting filter {filterno} values: {value1}, {value2}")
#     set_filter.testval = args[0]
#     testval = args[0]
#     # return testval


# dispatcher.map("/filter*", set_filter)  # Map wildcard address to set_filter function

# # Set up server and client for testing
# from pythonosc.osc_server import BlockingOSCUDPServer
# from pythonosc.udp_client import SimpleUDPClient

# server = BlockingOSCUDPServer(("127.0.0.1", 1337), dispatcher)
# client = SimpleUDPClient("127.0.0.1", 1337)

# # Send message and receive exactly one message (blocking)
# client.send_message("/filter1", [1., 2.])
# server.handle_request()

# client.send_message("/filter8", [6., -2.])
# server.handle_request()


#%% osc4py3 test:
# import logging

# import pandas as pd
# import numpy as np
# import gym
# import argparse
# import random
# import time

# from pythonosc import udp_client

# import reference_environment_direct_deployment

# from pathlib import Path

# # Import needed modules from osc4py3
# from osc4py3.as_eventloop import *
# from osc4py3 import oscmethod as osm

# def handlerfunction(s, x, y):
#     # Will receive message data unpacked in s, x, y
#     # pass
#     print(s)
#     return s

# def handlerfunction2(address, s, x, y):
#     # Will receive message address, and message data flattened in s, x, y
#     pass

# # Start the system.
# osc_startup()

# # Make server channels to receive packets.
# osc_udp_server("127.0.0.1", 3721, "aservername")
# # osc_udp_server("0.0.0.0", 3724, "anotherserver")

# # Associate Python functions with message address patterns, using default
# # argument scheme OSCARG_DATAUNPACK.
# osc_method("/test/*", handlerfunction)
# # Too, but request the message address pattern before in argscheme
# osc_method("/test/*", handlerfunction2, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATAUNPACK)

# # Periodically call osc4py3 processing method in your event loop.
# finished = False
# while not finished:
#     # …
#     osc_process()
#     # …

# # Properly close the system.
# osc_terminate()

