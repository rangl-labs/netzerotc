#!/usr/bin/env python

import os
import logging
from pathlib import Path
import numpy as np

import requests
import docker
from docker.types import Mount

from util import read_seeds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


client = docker.from_env()

# set up a submission (agent) container
# assumes an image "submission:v0.1.0" was created using `make` in reference/environment
seeds = read_seeds()
scores = []
for seed in seeds:
    submission = client.containers.run(
        image="submission:ags_v02",
        name="agent",
        network="local_rangl",
        detach=False,
        auto_remove=False,
        # command="sleep infinity",  # debug
        environment={
            "RANGL_SEED": seed,
            "RANGL_ENVIRONMENT_URL": "http://environment:5000",
        },
    )
    logger.debug(f"Created submission")
    logger.debug(f"Completed submission")

    # TODO evaluation script should be executed here, but while prototyping we do it here

    # fetch results from environment
    output = submission.decode("utf-8").strip()

    print("output")
    print(output)

    # assumption: final line in stdout is the instance id
    instance_id = output.split("\n")[-1]
    logger.debug(f"Instance id: {instance_id}")

    # send a request to the score endpoint
    ENVIRONMENT_URL = "http://localhost:5000/"

    # fetch score for submission
    url = f"{ENVIRONMENT_URL}/v1/envs/{instance_id}/score/"
    response = requests.get(f"{ENVIRONMENT_URL}/v1/envs/{instance_id}/score/").json()

    score = response["score"]["value1"]
    print("score:", score)
    scores.append(score)

print("Evaluation completed using {} seeds.".format(len(seeds)))
print("Final average score: ", np.average(scores))
