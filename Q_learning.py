import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999
MIN_EPSILON = .01
def default_Q_value():
    return 0


if __name__ == "__main__":



    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)
    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n
    Q_table = defaultdict(default_Q_value)# starts with a pessimistic estimate of zero reward for each state.
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    episode_reward_record = deque(maxlen=100)
    alpha = LEARNING_RATE
    for i in range(EPISODES):
        current_state = env.reset()
        done = False
        # sum the rewards that the agent gets from the environment
        total_episode_reward = 0
        while done == False:
            explore_probability = EPSILON
            if np.random.uniform(0, 1) < explore_probability:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[current_state, :])

            # The environment runs the chosen action and returns
            # the next state, a reward and true if the epiosed is ended.
            next_state, reward, done, info = env.step(action)
            nextMax = np.max(Q[next_state,:])
            # We update our Q-table using the Q-learning iteration
            newval  =Q_table[current_state, action] + alpha * (reward + DISCOUNT_FACTOR *nextMax -Q_table[current_state, action])
            Q_table[current_state,action] = newval
            Q[current_state,action] = newval
            total_episode_reward = total_episode_reward + reward
            # If the episode is finished, we leave the for loop
            if done:
                episode_reward_record.append(total_episode_reward)
                val =Q_table[current_state,action]+alpha*(reward- Q_table[current_state, action])
                Q_table[current_state, action] = val
                Q[current_state,action] = val

                break
            current_state = next_state
        EPSILON = EPSILON*EPSILON_DECAY
        if EPSILON < MIN_EPSILON:
            EPSILON= MIN_EPSILON
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )


    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################







