'''
The following code implements TD Control Algorithms: SARSA, Q-learning and Expected SARSA to Frozen Lake environment.

@Author: Nirav Raiyani
Department of Chemical and Materials Engineering
University of Alberta
'''

import gym
import numpy as np
import time
from tqdm import tqdm


############################### General Functions #######################################

def init_Q(s, a, type="zeros"):
    '''
    This function initializes the table of Action-value function for each state and action.
    :param s: No. of states
    :param a: NO. of possible action available
    :param type: "zeros", "Ones", "Random"
    :return: s x a dimensional matrix for action value function Q(s, a).
    '''
    if type == "ones":
        q = np.ones((s, a))

    if type == "zeros":
        q = np.zeros((s, a))

    if type == "random":
        q = np.random.random((s, a))

    return q


def e_greedy(no_a, e, q):
    """
    This function performs the epsilon greedy action selection
    :param no_a: No. of actions available
    :param e: Exploration parameter
    :param q: Action value function for the current state
    :return: epsilon greedy action
    """
    k = np.random.rand()
    if k < e:
        a = np.random.randint(0, no_a)
    else:
        a = np.argmax(q)
    return a


################## SARSA ###########################################
def SARSA(alpha, gamma, epsilon, episodes, max_step, n_tests, render=True, test=False):
    """
    Implimentation of SARSA to the Frozen Lake environment.
    :param alpha: learning rate
    :param gamma: Discounting
    :param epsilon: Exploration Parameter
    :param episodes: Number of episodes to run simulations
    :param max_step: Maximum no. of steps allowed for every episodes if it doesn't terminate
    :param n_tests: No of test runs
    :param render: True or False (For Visualization)
    :param test: True or False
    """

    env = gym.make('FrozenLake-v0')
    n_s, n_a = env.observation_space.n, env.action_space.n
    Q = init_Q(n_s, n_a, "ones")
    episode_reward_train = []
    for i in tqdm(range(episodes)):

        # Resetting the reward
        total_reward = 0

        # Resetting the environment
        s = env.reset()

        # Selecting a greedy action
        a = e_greedy(n_a, epsilon, Q[s, :])

        t = 0
        done = False
        while t < max_step:
            if render:
                env.render()
            # Taking a step into the environment
            next_s, reward, done, info = env.step(a)

            # Selecting a greedy action in the next state
            next_a = e_greedy(n_a, epsilon, Q[next_s, :])
            total_reward += reward

            # Updating the action value estimate based on the response of the environment
            if done:
                Q[s, a]  += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + gamma * Q[next_s, next_a] - Q[s, a])
            t += 1

            # Assigning next state as a current state
            s, a = next_s, next_a

            if done:
                if render:
                    env.render()
                    print("Episode {i} took {t} steps ")
                episode_reward_train.append(total_reward)
                break
    if test:
        episode_reward_test = test_agent(Q, env, n_tests, n_a)
    return episode_reward_train, episode_reward_test, Q


####################### Q -Learning #########################################
def TD_Q(alpha, gamma, epsilon, episodes, max_step, n_tests, render=True, test=False):
    env = gym.make('FrozenLake8x8-v0')
    n_s, n_a = env.observation_space.n, env.action_space.n
    Q = init_Q(n_s, n_a, "ones")
    episode_reward_train = []

    for i in tqdm(range(episodes)):
        t = 0
        total_reward = 0
        s = env.reset()
        a = e_greedy(n_a, epsilon, Q[s, :])

        while t < max_step:

            next_s, reward, done, info = env.step(a)
            next_a = e_greedy(n_a, epsilon, Q[next_s, :])
            max_a = np.argmax(Q[next_s,:])
            total_reward += reward
            if done:
                Q[s, a] += alpha*(reward  - Q[s, a])
            else:
                Q[s, a] += alpha*(reward + gamma*Q[next_s, max_a] - Q[s, a])
                # or
                # Q[s, a] += alpha*(reward + gamma*np.max(Q[next_s, :]) - Q[s, a])

            t += 1
            s, a = next_s, next_a

            if done:
                if render:
                    env.render()
                    print("Episode {i} took {t} steps ")
                episode_reward_train.append(total_reward)
                break

    if test:
        episode_reward_test = test_agent(Q, env, n_tests, n_a)
    return episode_reward_train, episode_reward_test, Q

############################    E - SARSA ####################################################

def expected_Q(Q, epsilon, n_a):

    a = np.argmax(Q)
    E_Q = np.sum(np.multiply((epsilon/n_a),Q)) + (1 - epsilon)*Q[a]

    return E_Q




def ESARSA(alpha, gamma, epsilon, episodes, max_step, n_tests, render=True, test=False):
    env = gym.make('FrozenLake-v0')
    n_s, n_a = env.observation_space.n, env.action_space.n
    Q = init_Q(n_s, n_a, "ones")
    episode_reward_train = []

    for i in tqdm(range(episodes)):
        t = 0
        total_reward = 0
        s = env.reset()
        a = e_greedy(n_a, epsilon, Q[s, :])

        while t < max_step:

            next_s, reward, done, info = env.step(a)
            next_a = e_greedy(n_a, epsilon, Q[next_s, :])
            total_reward += reward
            if done:
                Q[s, a] += alpha*(reward - Q[s, a])
            else:
                Q[s, a] += alpha*(reward + gamma*expected_Q(Q[next_s, :], epsilon, n_a) - Q[s, a])

            t += 1
            s, a = next_s, next_a

            if done:
                if render:
                    env.render()
                    print("Episode {i} took {t} steps ")
                episode_reward_train.append(total_reward)
                break

    if test:
        episode_reward_test = test_agent(Q, env, n_tests, n_a)
    return episode_reward_train, episode_reward_test, Q

def test_agent(q, env, n_test, n_a):
    episode_reward_test = []
    for test in range(n_test):
        s = env.reset()
        done = False

        # Making the target policy entirely greedy
        epsilon = 0
        total_reward = 0
        while True:
            time.sleep(0)
            env.render()

            # Taking a greedy action based on current state-value function
            a = np.argmax(q[s, :])

            # Taking an action and experiencing the response
            s, reward, done, info = env.step(a)

            total_reward += reward
            optimal = 0
            if s == 15:
                print("sucess")
                optimal += optimal

            if done:
                print("Episode ended")
                time.sleep(0.5)
                episode_reward_test.append(total_reward)
                break
    return episode_reward_test


if __name__ == "__main__":
    alpha = 0.1
    gamma = 1
    epsilon = 0
    episode = 6000
    max_steps = 2000
    n_test = 20
    train, test, Q = TD_Q(alpha, gamma, epsilon, episode, max_steps, n_test, render=False, test=True)
    print("Training Rewards:", train)
    print("Test Rewards:", test)
    sucess_rate = np.sum(test)/n_test
    print("sucess rate:", sucess_rate)


# # GYM environment has different transition probabilities for same state and same action
# # Proof
# import gym
# import numpy as np
# next_state = []
# env = gym.make("FrozenLake-v0")
# for i in range(25):
#     s = env.reset()
#     a = 1
#     s, reward, info, do = env.step(a)
#     next_state.append(s)
