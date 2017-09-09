""" Implementation of a Deep Q Network (DQN) in the setting of the CartPole_v0 environment

"""

__author__ = "Jan Krepl"
__email__ = "jankrepl@yahoo.com"
__license__ = "MIT"

import gym
from gym import wrappers
from foo import *

# PARAMETERS
# Environment
NUMBER_OF_EPISODES = 3500

# Neural Net
nn_parameter_dict = {'MEMORY_SIZE': 10000,
                     'BATCH_SIZE': 32,
                     'HL_1_SIZE': 5,
                     'HL_2_SIZE': 5,
                     'GAMMA': 1
                     }

# Policy
EPSILON = 1
EPSILON_DECAY = (0.001 / EPSILON) ** (1 / (NUMBER_OF_EPISODES - 1000))

# Testing - a test is an independent GREEDY run of our trained network
enable_test_mode = False
number_of_tests = 10
test_length = 10

# Recording and submission
bool_record = True
output_folder = ''

bool_submit = True
API_KEY = ''

# MAIN ALGORITHM
env = gym.make('CartPole-v0')
if bool_record:
    env = wrappers.Monitor(env, output_folder)

my_nn = Neural_Network(nn_parameter_dict)

test_schedule = list(map(int, np.linspace(0, NUMBER_OF_EPISODES, number_of_tests)))
print('The tests are planned for the following episodes: ' + str(test_schedule))



for e in range(NUMBER_OF_EPISODES):
    if e in test_schedule and enable_test_mode:
        # TEST MODE
        for e_test in range(test_length):
            s_old = env.reset()

            for t in range(201):
                #env.render()
                a = my_nn.choose_action(s_old, 0)  # BE GREEDY
                s_new, r, d, _ = env.step(a)
                s_old = s_new
                if d:
                    print('TEST: The episode ' + str(e_test) + ' lasted for ' + str(t))
                    break
    else:
        # TRAIN MODE
        s_old = env.reset()

        eps = EPSILON * (EPSILON_DECAY ** e)
        for t in range(201):
            env.render()
            a = my_nn.choose_action(s_old, eps)
            s_new, r, d, _ = env.step(a)
            my_nn.memorize(s_old, a, r, s_new, d)
            my_nn.train()
            s_old = s_new
            if d:
                print('TRAIN: The episode ' + str(e) + ' lasted for ' + str(t) + ' timesteps with epsilon ' + str(eps))
                my_nn.update_MRR(t)
                break

env.close()

if bool_submit:
    gym.upload(output_folder, api_key=API_KEY)

