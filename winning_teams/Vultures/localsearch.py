import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from itertools import count

ENV_NAME = "rangl:nztc-closed-loop-v0"
MOVING_AVERAGE = 30
NUM_EPISODES = 10000
DELTA = 0.1
MAX_LEVEL = [27, 25, 24]
MIN_SCORE = -5000000

scores = []
moving_averages = []

opt_policy = [0, 0, 0]*20
temp_policy = np.zeros(60)
best_score = MIN_SCORE

counter = 0


def plot_scores(show_ma=False, show_trend=False):
	plt.figure(2)
	plt.clf()
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Score')

	y = scores
	x = np.arange(len(y)) + 1

	moving_averages.append(np.mean(y[-MOVING_AVERAGE:]))

	plt.plot(x, y, label="Episode scores")

	if show_ma:
		plt.plot(x, moving_averages, linestyle="--", label="Last " + str(MOVING_AVERAGE) + " episodes average")


	if show_trend and len(x) > 1:
		z = np.polyfit(x, y, 1)
		p = np.poly1d(z)
		plt.plot(x, p(x), linestyle="-.", label="Trend")


	plt.legend(loc="upper left")
	plt.pause(0.001)


def evaluate_policy(policy):
	state = env.reset()

	for t in range(20):
		action = policy[3*t:3*(t+1)]
		next_state, reward, done, _ = env.step(action)

		if done:
			return sum(env.state.rewards_all)

		state = next_state


env = gym.make(ENV_NAME)
env.reset()


for episode in range(NUM_EPISODES):
	score = evaluate_policy(opt_policy)
	counter = counter + 1

	if (score > best_score):
		best_score = score
		print(score, opt_policy)
		counter = 0

	#scores.append(score)
	#plot_scores(show_ma=True, show_trend=True)

	best_local_plus = [MIN_SCORE]*60
	best_local_min = [MIN_SCORE]*60
	for i in range(60):
		temp_policy = opt_policy.copy()

		if (temp_policy[i] + DELTA < MAX_LEVEL[i % 3]):
			temp_policy[i] = temp_policy[i] + DELTA
			best_local_plus[i] = evaluate_policy(temp_policy)

	for i in range(60):
		temp_policy = opt_policy.copy()

		if (temp_policy[i] - DELTA > 0):
			temp_policy[i] = temp_policy[i] - DELTA
			best_local_min[i] = evaluate_policy(temp_policy)

	if (np.max(best_local_plus) > np.max(best_local_min)):
		max_index = np.argmax(best_local_plus)
		opt_policy[max_index] = opt_policy[max_index] + DELTA
	else:
		max_index = np.argmax(best_local_min)
		opt_policy[max_index] = opt_policy[max_index] - DELTA

	if (counter == 4):
		print("Done!")
		break

env.close()

