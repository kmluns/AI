# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:29:30 2019

@author: kmluns
"""

# %% all import

import gym
import gym_chess


import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D

# save model
from keras.models import model_from_json

#graph
import matplotlib.pyplot as plt


# %% class DQL Agent
class DQLAgent:
    def __init__(self,env):
        #hyperparams, params
        self.env = env
        self.state_shape = self.env.observation_space.shape
        
        self.action_size = self.env.action_space.n
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        
        self.memory = deque(maxlen = 100)
        
        self.model = self.build_model()
        self.target_model = self.build_model()
        
        self.random = 0
        self.action = 0
        self.notAction = 0
        self.forcedAction =0
        self.resign = 0
        
        
# =============================================================================
#         # serialize model to JSON
#         model_json = self.model.to_json()
#         with open("model/big/model.json", "w") as json_file:
#             json_file.write(model_json)
# =============================================================================


    
    def build_model(self):
        model = Sequential()
# =============================================================================
#         model.add(Conv1D(filters = 512, kernel_size = (5),padding = 'Same', 
#                          activation ='relu', input_shape = self.state_shape))
#         model.add(MaxPool1D(pool_size=(2)))
#         model.add(Dropout(0.25))
#         #
#         model.add(Conv1D(filters = 512, kernel_size = (3),padding = 'Same', 
#                          activation ='relu'))
#         model.add(MaxPool1D(pool_size=(2), strides=(2)))
#         model.add(Dropout(0.5))
#         model.add(Conv1D(filters = 128, kernel_size = (3),padding = 'Same', 
#                          activation ='relu', input_shape = self.state_shape))
#         model.add(MaxPool1D(pool_size=(2)))
#         model.add(Dropout(0.25))
#         model.add(Conv1D(filters = 128, kernel_size = (3),padding = 'Same', 
#                          activation ='relu', input_shape = self.state_shape))
#         model.add(Dropout(0.8))
#         model.add(Conv1D(filters = 64, kernel_size = (3),padding = 'Same', 
#                          activation ='relu', input_shape = self.state_shape))
#         model.add(Dropout(0.25))
# =============================================================================        
        # fully connected
        
        #model.add(Dense(256, activation='relu'))
        #model.add(Dropout(0.8))
        model.add(Dense(256, input_shape= self.state_shape, activation='relu'))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done ):
        self.memory.append((state, action, reward, next_state, done))
    
    def adaptiveEGready(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def select_action(self, state, player):  
        resign = False
        actions = self.env.get_possible_actions(self.env.state, player)
        if len(actions) == 0:
            self.resign +=1
            resign = True
            return self.env.resign_action(), resign
        # acting : explore or exploit
        if random.uniform(0,1) <= self.epsilon:
            #explore
            self.random += 1
            return random.choice(actions), resign
        # exploit
        act_values = self.model.predict(state.reshape(-1,8,8))
        #predicted_actions = act_values[0]
        #action_index = np.argmax(predicted_actions[actions])
        #return actions[action_index]
        action = np.argmax(act_values[0])
        if action in actions:
            self.action += 1
        else:
            if self.notAction > self.forcedAction /2:
                predicted_actions = act_values[0]
                action_index = np.argmax(predicted_actions[actions])
                action = actions[action_index]
                self.forcedAction += 1
            else:
                self.notAction += 1
        return action, resign
        
    def replay(self,batch_size):
        "vectorized replay method"
        if len(agent.memory) < batch_size:
            return
        # Vectorized method for experience replay
        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])

        # If minibatch contains any non-terminal states, use separate update rule for those states
        if len(not_done_indices[0]) > 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))
            
            # Non-terminal update rule
            y[not_done_indices] += np.multiply(self.gamma, predict_sprime_target[not_done_indices, np.argmax(predict_sprime[not_done_indices, :][0], axis=1)][0])

        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(batch_size), actions] = y
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=1, verbose=0)
        
        
# =============================================================================
#     def replay(self, batch_size):
#         # training
#         if len(self.memory) < batch_size:
#             return
#         minibatch = random.sample(self.memory,batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             if done:
#                 target = reward 
#             else:
#                 #next_state = swapped = np.moveaxis(next_state, 0, 2)
# #                next_state = next_state.reshape(-1,210,160,3)
#                 target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
#             
# #            state = state.reshape(-1,210,160,3)
#             train_target = self.model.predict(state)
#             train_target[0][action] = target
#             self.model.fit(state,train_target, verbose = 1)
# =============================================================================
            
        
    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save_weights("model/big/target_model_acc.h5")
        print("action : {}, notaction : {}, random : {}, forcedAction : {}, resign : {} ".format(self.action, self.notAction, self.random, self.forcedAction, self.resign))
        self.random = 0
        self.action = 0
        self.notAction = 0
        self.forcedAction = 0
        self.resign = 0

        
    def model_save_best(self, reward):
        self.model.save_weights("model/big/best_model"+reward+".h5")
        self.target_model.save_weights("model/big/best_target_model"+reward+".h5")
        
    def model_load(self,filePath):
        self.model.load_weights(filePath)
        self.target_model.load_weights(filePath)


# %% create objects
env = gym.make('ChessVsRandomBot-v0')
agent = DQLAgent(env)


#agent.model_load("model/target_model_acc.h5")

# %% main function
if __name__ == "__main__":
    
    # initialize env and agent
    batch_size = 64
    
    
    player = 1
    
    
    best_total_reward = 0
    episodes = 10000
    
    rewards = []
    
    plt.plot(rewards)
    plt.ylabel('rewards')
    plt.show()
    
    for e in range(episodes):
        
        # initialize environment
        state = env.reset()
        state = state['board'].reshape(-1,8,8)
#        state = np.reshape(state,[1,4])
        
        
        total_reward = 0
        step_number = 0
        
        
        while True:
            #env._render()

            # act
            action, resign = agent.select_action(state, player)
            
            actions = env.get_possible_actions(env.state, player)
            
            next_state, reward, done = state, -1000, False
            
            if resign:
                reward, done = -100, True
                total_reward += reward
            
            elif action in actions:
                #print("not in action : {}".format(action))
                      
                # step
                next_state, reward, done, _ = env.step(action)
                if reward == 0:
                    reward = -0.0001
                next_state = next_state['board'].reshape(-1,8,8)
                total_reward += reward
            
            
            # remember / storage
            agent.remember(state, action, reward, next_state, done)
            
            # update state
            state = next_state
            
            # replay
            agent.replay(batch_size)
            
            # adjust epsilon
            agent.adaptiveEGready()
            
            step_number += 1
            
            if best_total_reward < total_reward:
                best_total_reward = total_reward
# =============================================================================
#                 agent.model_save_best(str(int(best_total_reward)))
# =============================================================================
            
            if done or total_reward < -100 or step_number > 500:
                print('Episode {}, reward {}, step {}'.format(e,total_reward, step_number))
                agent.targetModelUpdate()
                rewards.append(total_reward)
                break
            
            
        plt.plot(rewards)
        plt.pause(0.05)
                
                
# %% test
                    
# =============================================================================
# import time
# 
# trained_model = agent
# state = env.reset()
# state = np.reshape(state,[1,8])
# 
# time_t = 0
# while True:
#     env.render()
#     action = trained_model.select_action(state)
#     next_state, reward, done, _ = env.step(action)
#     next_state = np.reshape(next_state,[1,8])
#     state = next_state
#     time_t += 1
#     print(time_t)
#     #`time.sleep(0.05)
#     if done:
#         break
#     
#     
# print("Done!")
#                     
# =============================================================================
    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
 