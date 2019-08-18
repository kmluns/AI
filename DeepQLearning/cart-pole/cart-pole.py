# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:29:30 2019

@author: kmluns
"""

# %% all import
import numpy as np
import gym
import random
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Dropout


# %% class DQL Agent
class DQLAgent:
    def __init__(self,env):
        #hyperparams, params
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.memory = deque(maxlen = 4000)
        
        self.model = self.build_model()
        self.target_model = self.build_model()


    
    def build_model(self):
        # neural network for deep q learning
        optimizer= Adam(lr = self.learning_rate)
        
        model = Sequential()
        model.add(Dense(64,input_dim = self.state_size, activation = "relu"))
        model.add(Dense(64, activation = "relu"))
        model.add(Dense(self.action_size, activation = "linear"))
        model.compile(loss = "mse", optimizer= optimizer)
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def adaptiveEGready(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def select_action(self,state):
        # acting : explore or exploit
        if random.uniform(0,1) <= self.epsilon:
            return self.env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
        
        
    def replay(self,batch_size):
        "vectorized replay method"
        if len(self.memory) < batch_size:
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
            
        
    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.save_weights("model/target_model.h5")
        
    def model_save_best(self):
        self.model.save_weights("model/best_model.h5")
		

# %% create objects
env = gym.make('CartPole-v1')
agent = DQLAgent(env)


# %% main function
if __name__ == "__main__":
    
    # initialize env and agent
    batch_size = 64
    
    
    
    best_total_reward = 0
    episodes = 10000
    for e in range(episodes):
        
        # initialize environment
        state = env.reset()
        state = np.reshape(state,[1,4])
        
        
        total_reward = 0
        
        while True:
            env.render()

            # act
            action = agent.select_action(state)
                      
            # step
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state,[1,4])
            
            total_reward += reward
            
            # remember / storage
            agent.remember(state, action, reward, next_state, done)
            
            # update state
            state = next_state
            
            # replay
            agent.replay(batch_size)
            
            # adjust epsilon
            agent.adaptiveEGready()
            
            if done or total_reward < -100:
                print('Episode {}, time {}'.format(e,total_reward))
                agent.targetModelUpdate()
                break
            
            if best_total_reward < total_reward:
                best_total_reward = total_reward
                agent.model_save_best()
            
                
                
# %% test
                    
import time

trained_model = agent
state = env.reset()
state = np.reshape(state,[1,8])

time_t = 0
while True:
    env.render()
    action = trained_model.select_action(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state,[1,8])
    state = next_state
    time_t += 1
    print(time_t)
    #`time.sleep(0.05)
    if done:
        break
    
    
print("Done!")
                    
    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
 