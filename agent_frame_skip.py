from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import SGD, Adam
from collections import deque

import numpy as np
import random
import cv2
import gym
import matplotlib.pyplot as plt
import os
import psutil

process = psutil.Process(os.getpid())
EPOCHS = 400
buffer = deque(maxlen=4)

class AgentDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.GAMMA = 0.95
        self.EPSILON = 1.0
        self.EPSILON_MIN = 0.1
        self.EPSILON_DECAY = 0.99
        self.LEARNING_RATE = 0.00001
        self.MOMENTUM = 0.9
        self.LEARNING_RATE_DECAY = 0.9
        self.BATCH_SIZE = 32
        self.optimizer = SGD(self.LEARNING_RATE, momentum=self.MOMENTUM, decay=self.LEARNING_RATE_DECAY)
        self.model = self.create_model()

        self.loss_history = []
        self.iteration_history = []


    def create_model(self):
        model = Sequential()
        model.add(Conv2D(16, 8, strides=4, input_shape=(84, 84, 4), activation='relu'))
        model.add(Conv2D(32, 4, strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(env.action_space.n, activation='linear'))
        model.compile(loss="mse", optimizer=self.optimizer)
        return model


    def act(self, state):
        rand = random.random()
        if rand <= self.EPSILON:
            return random.randint(0, self.action_size - 1)
        else:
            nn_out = self.model.predict(state)
            return np.argmax(nn_out[0])


    def remember(self, expirience):
        self.memory.append(expirience)


    def replay(self, iteration):
        if len(self.memory) < self.BATCH_SIZE:
            return

        batch = np.array(random.sample(self.memory, self.BATCH_SIZE))

        states = np.array([s[0] for s in np.array(batch[:, 0])])
        actions = np.array(batch[:, 1], dtype='uint8')
        rewards = batch[:, 2]
        next_states = np.array([state[0] for state in np.array(batch[:, 3])])
        done = batch[:, 4]
        mask = np.array([not d for d in done])

        q = self.model.predict(states)
        next_q = self.model.predict(next_states)
        q[mask, actions[mask]] = rewards[mask] + self.GAMMA * np.amax(next_q[mask], axis=1)
        history = self.model.fit(states, q, epochs=1, verbose=0)
        self.loss_history.append(sum(history.history['loss']) / len(history.history['loss']))


    def epsilon_decay(self):
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY


    def get_avg_loss(self):
        if len(self.loss_history) > 0:
            x = sum(self.loss_history) / len(self.loss_history)
        else:
            x = -1
        self.loss_history = []
        return x


def init_memory(env, agent):
    env.reset()
    for i in range(1000):
        if len(buffer) == 4:
            last_state = np.array(buffer)
            action = random.randint(0, env.action_space.n - 1)
            frame, reward, done, info = env.step(action)
            frame = preprocess_frame(frame)
            buffer.append(frame)
            next_state = np.array(buffer)

            agent.remember([
                last_state.reshape(1,84,84,4),
                action,
                reward,
                next_state.reshape(1,84,84,4),
                done])

        else:
            action = random.randint(0, env.action_space.n-1)
            frame, reward, done, info = env.step(action)
            frame = preprocess_frame(frame)
            buffer.append(frame)



def preprocess_frame(frame):
    frame = cv2.resize(frame, (84, 84))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame



if __name__ == "__main__":
    env = gym.make("SpaceInvadersDeterministic-v4")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = AgentDQN(state_size, action_size)
    init_memory(env, agent)
    print("Init mem done...")

    try:
        for i in range(EPOCHS):
            env.reset()
            iteration_counter = 1
            lives = 3
            total_reward = 0
            while True:
                state = np.array(buffer).reshape(1,84,84,4)
                action = agent.act(state)
                frame, reward, done, info = env.step(action)
                frame = preprocess_frame(frame)
                buffer.append(frame)
                next_state = np.array(buffer).reshape(1,84,84,4)

                if info["ale.lives"] < lives: 
                    reward = -5
                    lives = info["ale.lives"]

                total_reward += reward

                agent.remember([state, action, reward, next_state, done])
                agent.replay(iteration_counter)

                # if agent.EPSILON < agent.EPSILON_MIN and EPOCHS > 50:
                #   env.render()

                if done:
    

                    print("EPOCH {} END. Iterations: {} Avg. loss: {} Reward: {} EPSILON: {} Memory: {} MB".format(
                        i,
                        iteration_counter,
                        agent.get_avg_loss(),
                        total_reward,
                        agent.EPSILON,
                        round(process.memory_info().rss * 10**(-6)), 2))
                    break
                iteration_counter += 1
            agent.epsilon_decay()
    except KeyboardInterrupt:
        pass

    agent.model.save("model.h5")
    plt.plot(agent.loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()