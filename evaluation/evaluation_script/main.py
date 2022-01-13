import json

# import random

import docker
import requests

ENVIRONMENT_URL = "http://nztc:5000"


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    print("Starting Evaluation.....")
    print("evaluation_script/main.py")
    print("Submission related metadata:")

    # random.seed(3423232)  # set a seed so that we generate the same seed_list each time
    # N_seeds = 5
    # seed_list = [random.randint(0, 1e7) for _ in range(N_seeds)]

    # explicitly listing the seeds from https://github.com/rangl-labs/netzerotc/blob/main/local_agent_training_and_evaluation/seeds.csv as below:
    seed_list = [
        6363661,
        1940112,
        9650219,
        2890621,
        8651466,
        1834208,
        7986844,
        7053405,
        # 6612160,
        # 9156319,
        # 6333356,
        # 133106,
        # 314808,
        # 5623388,
        # 221975,
        # 1504387,
        # 6970805,
        # 7341740,
        # 7576412,
        # 8297526,
        # 571960,
        # 5459592,
        # 2734582,
        # 9536951,
        # 9716755,
        # 2018774,
        # 7876799,
        # 159754,
        # 4077557,
        # 3571375,
        # 164906,
        # 3681712,
        # 4703580,
        # 4666427,
        # 3298598,
        # 1534356,
        # 6531179,
        # 7759559,
        # 6815011,
        # 3911076,
        # 7942634,
        # 3715938,
        # 5198377,
        # 9535630,
        # 2166600,
        # 6854299,
        # 1134615,
        # 1175505,
        # 7386101,
        # 9921565,
        # 2069437,
        # 8558157,
        # 9230516,
        # 4624815,
        # 6512722,
        # 8281745,
        # 5897237,
        # 1925185,
        # 6901224,
        # 6934016,
        # 4004741,
        # 8041193,
        # 6667510,
        # 3720438,
        # 4884870,
        # 9873136,
        # 7773061,
        # 7630923,
        # 7115293,
        # 9724079,
        # 6028391,
        # 882684,
        # 8476098,
        # 4595056,
        # 4698906,
        # 732723,
        # 6337448,
        # 6382589,
        # 7010890,
        # 9936259,
        # 4300625,
        # 6426103,
        # 5202352,
        # 823788,
        # 8368708,
        # 8891441,
        # 6171686,
        # 1351259,
        # 1624573,
        # 9537083,
        # 7355588,
        # 3361680,
        # 361488,
        # 9307853,
        # 3703568,
        # 5044473,
        # 377657,
        # 8930331,
        # 2121375,
        # 7156888,
    ]

    metadata = kwargs["submission_metadata"]
    print(metadata)

    # Create docker client and login to AWS elastic container registry (ECR)
    print("Creating docker client")
    client = docker.DockerClient(base_url="unix://var/run/docker.sock")

    # EXAMPLES:
    # input_file_relative:
    # /media/submission_files/submission_81/2a099d6e-63c5-474c-9764-a6b593889e6a.json
    # input_file_full:
    # /code/media/submission_files/submission_81/2a099d6e-63c5-474c-9764-a6b593889e6a.json
    input_file_relative = metadata["input_file"]
    input_file_full = f"/code{input_file_relative}"

    with open(input_file_full) as json_file:
        data = json.load(json_file)
        submitted_image_uri = data["submitted_image_uri"]

    print(f"submitted_image_uri: {submitted_image_uri}")

    score_list = []
    for seed in seed_list:
        container_output = client.containers.run(
            submitted_image_uri,
            network="evalai_rangl",
            environment=[
                "RANGL_ENVIRONMENT_URL=http://nztc:5000/",
                f"RANGL_SEED={seed}",
            ],
        )
        output = container_output.decode("utf-8").strip()

        print("output")
        print(output)

        # assumption: final line in stdout is the instance id
        instance_id = output.split("\n")[-1]
        print(f"instance_id: {instance_id}")

        # fetch score for submission
        ENVIRONMENT_URL = "http://nztc:5000/"
        url = f"{ENVIRONMENT_URL}/v1/envs/{instance_id}/score/"
        response = requests.get(
            f"{ENVIRONMENT_URL}/v1/envs/{instance_id}/score/"
        ).json()

        print(response)
        score = float(response["score"]["value1"])
        score_list.append(score)

    mean_score = sum(score_list) / len(score_list)
    print("mean_score", mean_score)

    # TODO: exctract score from response
    # TODO: return the score in output below

    print("phase_codename", phase_codename)

    output = {}
    output["result"] = [
        {
            "test_split": {
                "Average Cost": mean_score,
            }
        }
    ]
    output["submission_result"] = output["result"][0]["test_split"]
    print("output", output)
    print(f"Completed evaluation for {phase_codename} phase")

    return output
