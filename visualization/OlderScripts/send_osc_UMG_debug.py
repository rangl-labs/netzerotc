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

# Specify the OSC (in Unreal Engine) server address

client_toUMGdebug = SimpleUDPClient("127.0.0.1", 6006)
client_toUMGdebug.send_message("/debugUMG", np.array([0.01, 99.9, 20.0]))

