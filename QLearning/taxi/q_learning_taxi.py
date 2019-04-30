# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 07:44:40 2019

@author: kmluns
"""

# %% imports
import gym
import numpy as np
import matplotlib.pyplot as plt
import time

# %%  
env = gym.make('Taxi-v2').env

# %% Q Table
q_table = np.zeros([env.observation_space.n, env.action_space.n])


# %% HyperParams
alpha = 0.1
gamma = 0.9
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.995


# %% Plotting Matrix
reward_list = []
dropouts_list = []

# %% episode

episode_number = 10000

for i in range(1,episode_number):
    
    # initialize environment
    state = env.reset()
    
    reward_count = 0
    dropouts = 0
    game_time = 0
    
    
    while True:
        
        game_time += 1
        
        # exploit vs explore to find action
        if(random.uniform(0,1) <= epsilon):
            action = env.action_space.sample()
        else:
             action = np.argmax(q_table[state])
             
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay

        
        # action process and take reward/observation
        next_state, reward, done, _ = env.step(action)
        
        
        # Q Learning function
        
        # old value
        old_value = q_table[state,action]
        
        # next_max
        next_max = np.max(q_table[next_state])
        
        # q learning 
        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        # update Q Table
        q_table[state,action] = next_value
        
        # update state
        state = next_state
        
        # find wrong dropouts
        if reward == -10:
            dropouts += 1
        
        reward_count += reward

        if done:
            break
        
# =============================================================================
#         if game_time > 10000:
#             break
# =============================================================================
        
    dropouts_list.append(dropouts)
    reward_list.append(reward_count)
    if i%10 == 0 :
        print('Episode: {}, reward {}, wrong dropout {}'.format(i,reward_count,dropouts))
        
        
        
        
        
        
        

# %%
        
        
pred_env = gym.make('Taxi-v2').env

score = 0
game_time = 0

state = pred_env.reset()


while True:
    game_time += 1
    
    # action
    action = np.argmax(q_table[state])
    
    # action process and take reward/observation
    next_state, reward, done, _ = pred_env.step(action)

    # update state
    state = next_state
    
    score += reward
    
    pred_env.render()
    print('Action: {}, state {}'.format(action,state))
    print('Time : {}, Score {}'.format(game_time,score))


    
    if done:
        break
    else:
        time.sleep(1)

        

        
        
        
        
        
        
        
        
        
        
        
        
        
        