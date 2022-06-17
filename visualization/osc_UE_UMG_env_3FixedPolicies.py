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

# define the event dispatcher for python-osc server; more details and explanation: https://github.com/attwad/python-osc and https://python-osc.readthedocs.io/
dispatcher = Dispatcher()

## initialize user specified actions, steps to jump back and move forward (deprecated, as here we only visualize the 3 scenarios)
# user_action = np.zeros(3)
# steps_back = 0
# steps_to_apply = 0
user_policy = 1 # initialize the user specified scenarios/policies number: 0 for breeze, 1 for gale, 2 for storm
user_year = 2031 # initialize the 1st year to apply action from the 3 possible scenarios, which is 2031

# define the function to be associated to the dispatcher in python-osc server; more details and explanation: https://github.com/attwad/python-osc and https://python-osc.readthedocs.io/
def set_user_action(address: str, *args: List[Any]) -> None:
    ## Set the variables below to be global, such that they can be changed by this function after receiving OSC message:
    # global steps_back
    # global user_action
    # global steps_to_apply
    global user_policy
    global user_year
    # We expect three float arguments
    # if not len(args) == 3 or type(args[0]) is not float or type(args[1]) or type(args[2]) is not float:
    #     return

    # Check that address starts with user_action
    if not address[:-1] == "/user_action":  # Cut off the last character
        return

    # value1 = args[1]
    # value2 = args[2]
    # value3 = args[3]
    user_policy = args[1]
    if user_policy == 0:
        user_policy_name = "Breeze"
    elif user_policy == 1:
        user_policy_name = "Gale"
    elif user_policy == 2:
        user_policy_name = "Storm"
    user_year = np.int64(args[0])
    # current_step = env.state.step_count
    # current_year = current_step + 2031
    current_year = user_year
    # current_step = address[-1]
    # print(f"\n \n Setting user_action values of step {current_step} to: [{value1}, {value2}, {value3}]")
    # print("\n\nCurrent env.state.step_count is %d; Setting user_action values of next step %d to: [%f, %f, %f]" %(current_step, current_step+1, value1, value2, value3))
    # set_user_action.steps_back = np.int64(abs(args[0]))
    # set_user_action.values = np.array([args[1], args[2], args[3]])
    # steps_back = np.int64(abs(args[0]))
    # user_action = np.array([args[1], args[2], args[3]])
    # steps_to_apply = np.int64(args[0])
    # if not steps_back:
    #     print("\n\nSetting env.state to step from env.state.step_count = %d to %d, after setting user_action values of next step %d to: [%f, %f, %f] ..." %(current_step, current_step+1, current_step+1, value1, value2, value3))
    # else:
    #     print("\n\nRewinding env.state from env.state.step_count = %d back to %d ..." %(current_step, current_step-steps_back))
    # if steps_to_apply > 0:
    #     print("\n\nSetting env.state to step from env.state.step_count = %d to %d, after setting user_policy number of these steps to %d ..." %(current_step, current_step + steps_to_apply, user_policy))
    # elif steps_to_apply < 0:
    #     print("\n\nRewinding env.state from env.state.step_count = %d back to %d ..." %(current_step, current_step + steps_to_apply))
    print("\n\nSetting env.state to user specified year = %d, after setting user_policy number to %d ..." %(current_year, user_policy))

dispatcher.map("/user_action*", set_user_action)  # Map wildcard address to set_user_action functionm; ore details and explanation: https://github.com/attwad/python-osc and https://python-osc.readthedocs.io/


# Specify the OSC server address (hosted by the "OSCMessageRender" blueprint in Unreal Engine)
ip = "127.0.0.1"
port = 5005
client_toUEosc = SimpleUDPClient(ip, port)  # Create OSC client
# Create another OSC client to send to the other OSC server hosted by the "MultiWindow_SlideYearsAndSwitch3Policies_UserReceivesOSC" blueprint in Unreal Engine
client_toUMGMultiWindow = SimpleUDPClient("127.0.0.1", 1336)

# Specify the OSC client and server address
# client = SimpleUDPClient("127.0.0.1", 1337) # deprecated; previously used to send user actions and step count back to Unreal Engine
# This is the server hosted by this script to listen to OSC message sent from UMG Widget "UMG_MultiWindow_UserSlideYearsAndSwitch3PoliciesOSC" in Unreal Engine:
server = BlockingOSCUDPServer(("127.0.0.1", 1338), dispatcher)


# Create environments
env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
# env_breeze = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
# env_gale = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
# env_storm = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
# set the seed equal to current time in rounded seconds
# seed = int(time.time())
seed = 123456 # use a fixed seed; can change to other integers
env.seed(seed)
# env_breeze.seed(seed)
# env_gale.seed(seed)
# env_storm.seed(seed)
# Reset the environments
env.reset()
# env_breeze.reset()
# env_gale.reset()
# env_storm.reset()

done = False
# done_Breeze = False
# done_Gale = False
# done_Storm = False
# rewards = []
# deployments = np.array(np.array(pd.read_excel('STORM_Deployments_Modified.xlsx'))[-(env.param.steps_per_episode+1):,1:],dtype=np.float32)
# Read the deployments numbers from the 3 scenarios:
deployments_Breeze = np.array(np.array(pd.read_excel('BREEZE_Deployments_Modified.xlsx'))[-(env.param.steps_per_episode+1):,1:],dtype=np.float32)
deployments_Gale = np.array(np.array(pd.read_excel('GALE_Deployments_Modified.xlsx'))[-(env.param.steps_per_episode+1):,1:],dtype=np.float32)
deployments_Storm = np.array(np.array(pd.read_excel('STORM_Deployments_Modified.xlsx'))[-(env.param.steps_per_episode+1):,1:],dtype=np.float32)
# actions = deployments[1:,:] - deployments[:-1,:]

# Apply actions of Breeze, and then record the reward and emission:
actions = deployments_Breeze[1:,:] - deployments_Breeze[:-1,:]
while not env.state.is_done():
    observation, reward, _, _ = env.step(actions[env.state.step_count + 1])
rewards_Breeze = np.array(env.state.rewards_all)
emissions_Breeze = np.array(env.state.emission_amount_all)
# Plot the figure and save to Unreal Engine's project file (only needed when the random seed is changed above)
# env.plot("./UE_Projects/BasicVisualization1/Content/Figures/Breeze.png")

# Apply actions of Gale, and then record the reward and emission:
env.seed(seed)
env.reset()
actions = deployments_Gale[1:,:] - deployments_Gale[:-1,:]
while not env.state.is_done():
    observation, reward, _, _ = env.step(actions[env.state.step_count + 1])
rewards_Gale = np.array(env.state.rewards_all)
emissions_Gale = np.array(env.state.emission_amount_all)
# Plot the figure and save to Unreal Engine's project file (only needed when the random seed is changed above)
# env.plot("./UE_Projects/BasicVisualization1/Content/Figures/Gale.png")

# Apply actions of Storm, and then record the reward and emission:
env.seed(seed)
env.reset()
actions = deployments_Storm[1:,:] - deployments_Storm[:-1,:]
while not env.state.is_done():
    observation, reward, _, _ = env.step(actions[env.state.step_count + 1])
rewards_Storm = np.array(env.state.rewards_all)
emissions_Storm = np.array(env.state.emission_amount_all)
# Plot the figure and save to Unreal Engine's project file (only needed when the random seed is changed above)
# env.plot("./UE_Projects/BasicVisualization1/Content/Figures/Storm.png")


# Since done is fixed to be False, the following will server until the script is killed/closed
while not done:
    server.handle_request() # start to listen the OSC message sent from UMG Widget in Unreal Engine; the received OSC message will change global variables user_policy and user_year
    if user_policy == 0: # if user selected Breeze:
        print("\nValues sent to OSC server in Unreal Engine for visualization are [step_count, reward, 3-element deployments, CO2 emission amount]: " + str(np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount))))
        # Deprecated: dynamically plotting and saving figures step-by-step to Unreal Engine; no longer needed due to performance issue
        # env_breeze.plot("./UE_Projects/BasicVisualization1/Content/Figures/current_policy.png")
        client_toUEosc.send_message(
            "/some/address", 
            np.hstack((user_year-2031, 
                       rewards_Breeze[user_year-2031], 
                       deployments_Breeze[1:][user_year-2031], 
                       emissions_Breeze[user_year-2031]))
            ) # send user specified year's: step_count, Breeze's rewards, deployment numbers, and emission to Unreal Engine for visualization
        client_toUMGMultiWindow.send_message(
            "/some/address", 
            np.hstack((user_year-2031, 
                       rewards_Breeze[user_year-2031], 
                       deployments_Breeze[1:][user_year-2031], 
                       emissions_Breeze[user_year-2031]))
            )
    elif user_policy == 1: # if user selected Gale:
        print("\nValues sent to OSC server in Unreal Engine for visualization are [step_count, reward, 3-element deployments, CO2 emission amount]: " + str(np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount))))
        # Deprecated: dynamically plotting and saving figures step-by-step to Unreal Engine; no longer needed due to performance issue
        # env_gale.plot("./UE_Projects/BasicVisualization1/Content/Figures/current_policy.png")
        client_toUEosc.send_message(
            "/some/address", 
            np.hstack((user_year-2031, 
                       rewards_Gale[user_year-2031], 
                       deployments_Gale[1:][user_year-2031], 
                       emissions_Gale[user_year-2031]))
            ) # send user specified year's: step_count, Gale's rewards, deployment numbers, and emission to Unreal Engine for visualization
        client_toUMGMultiWindow.send_message(
            "/some/address", 
            np.hstack((user_year-2031, 
                       rewards_Gale[user_year-2031], 
                       deployments_Gale[1:][user_year-2031], 
                       emissions_Gale[user_year-2031]))
            )
    elif user_policy == 2: # if user selected Storm:
        print("\nValues sent to OSC server in Unreal Engine for visualization are [step_count, reward, 3-element deployments, CO2 emission amount]: " + str(np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount))))
        # Deprecated: dynamically plotting and saving figures step-by-step to Unreal Engine; no longer needed due to performance issue
        # env_storm.plot("./UE_Projects/BasicVisualization1/Content/Figures/current_policy.png")
        client_toUEosc.send_message(
            "/some/address", 
            np.hstack((user_year-2031, 
                       rewards_Storm[user_year-2031], 
                       deployments_Storm[1:][user_year-2031], 
                       emissions_Storm[user_year-2031]))
            ) # send user specified year's: step_count, Storm's rewards, deployment numbers, and emission to Unreal Engine for visualization
        client_toUMGMultiWindow.send_message(
            "/some/address", 
            np.hstack((user_year-2031, 
                       rewards_Storm[user_year-2031], 
                       deployments_Storm[1:][user_year-2031], 
                       emissions_Storm[user_year-2031]))
            )




#%% older implementation for user to manually input years to jump back or forward and apply any one of the 3 scenarios:
# while not done:
#     # action = actions[env.state.step_count + 1]
#     # client.send_message("/agent_action1", np.hstack((done, action, env.state.step_count + 1)))
#     client.send_message("/agent_action1", np.hstack((done, np.float(user_policy), env.state.step_count)))
#     server.handle_request()
#     if user_policy == 0:
#         actions = deployments_Breeze[1:,:] - deployments_Breeze[:-1,:]
#     elif user_policy == 1:
#         actions = deployments_Gale[1:,:] - deployments_Gale[:-1,:]
#     elif user_policy == 2:
#         actions = deployments_Storm[1:,:] - deployments_Storm[:-1,:]
#     if steps_to_apply < 0:
#         previous_actions = env.state.actions_all[:-abs(steps_to_apply)]
#         env.seed(seed)
#         env.reset()
#         for previous_action in previous_actions:
#             observation, reward, done, _ = env.step(previous_action)
#     elif steps_to_apply > 0:
#         # observation, reward, done, _ = env.step(user_action)
#         for i in np.arange(steps_to_apply):
#             observation, reward, done, _ = env.step(actions[env.state.step_count + 1])
#     # observation, reward, done, _ = env.step(user_action)
#     print("\nValues sent to OSC server in Unreal Engine for visualization are [step_count, reward, 3-element deployments, CO2 emission amount]: " + str(np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount))))
#     env.plot("./UE_Projects/BasicVisualization1/Content/Figures/current_state.png")
#     client_toUEosc.send_message("/some/address", np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount)))
#     client_toUMGpopup.send_message("/some/address", np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount)))
#     # client.send_message("/some/address", env.render())
#     if done:
#         # client.send_message("/agent_action1", np.hstack((done, action, env.state.step_count)))
#         client.send_message("/agent_action1", np.hstack((done, np.float(user_policy), env.state.step_count)))


# # def storm_agent(self, seeds):
# #     rewards = []
# #     # deployments = np.array(np.array(pd.read_excel('STORM_Deployments.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
# #     deployments = np.array(np.array(pd.read_excel('STORM_Deployments_Modified.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
# #     actions = deployments[1:,:] - deployments[:-1,:]
# #     for seed in seeds:
# #         self.env.seed(seed)
# #         self.env.reset()
# #         while not self.env.state.is_done():
# #             action = actions[self.env.state.step_count + 1]
# #             self.env.step(action)
# #         rewards.append(sum(self.env.state.rewards_all))
# #     return np.mean(rewards)



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

