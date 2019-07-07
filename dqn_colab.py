from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from collections import deque
from copy import copy

from google.colab import drive
drive.mount('/content/gdrive')


import numpy as np
import random
import cv2
import gym
import matplotlib.pyplot as plt
import os
import psutil

process = psutil.Process(os.getpid())
EPOCHS = 10000
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
        self.LEARNING_RATE = 0.000005
        self.BATCH_SIZE = 32
        self.optimizer = SGD(self.LEARNING_RATE, momentum=0.005)
        self.model = self.create_model()

        self.loss_history = []
        self.iteration_history = []

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, 8, strides=4, input_shape=(4, 84, 84), data_format="channels_first", activation='relu'))
        model.add(Conv2D(64, 4, strides=2, activation='relu'))
        model.add(Conv2D(64, 3, strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(env.action_space.n, activation='linear'))
        model.compile(loss="mse", optimizer=RMSprop(lr=0.00025,
                                                    rho=0.95, epsilon=0.01), )
        return model

    def act(self, state):
        rand = random.random()
        if rand <= self.EPSILON:
            return random.randint(0, self.action_size - 1)
        else:
            nn_out = self.model.predict(state)
            # print("NETWORK DECIDES", np.argmax(nn_out[0]))
            return np.argmax(nn_out[0])

    def remember(self, expirience):
        self.memory.append(expirience)

    def replay(self, iteration):
        if len(self.memory) < self.BATCH_SIZE:
            return

        batch = np.array(random.sample(self.memory, self.BATCH_SIZE))

        states = np.zeros(shape=(len(batch), 4, 84, 84))
        actions = np.zeros(len(batch), dtype='uint8')
        rewards = np.zeros(len(batch))
        next_states = np.zeros(shape=(len(batch), 4, 84, 84))
        mask = np.zeros(len(batch), dtype='bool')

        for i, b in enumerate(batch):
            states[i] = np.array(b[0])
            actions[i] = b[1]
            rewards[i] = b[2]
            next_states[i] = np.array(b[3])
            mask[i] = b[4]

        q = self.model.predict(states)
        next_q = self.model.predict(next_states)
        for i in range(len(q)):
            if mask[i]:
                q[actions[i]] = rewards[i]
            else:
                q[actions[i]] = rewards[i] + self.GAMMA * np.amax(next_q[i])

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
            last_state = copy(buffer)
            action = random.randint(0, env.action_space.n - 1)
            frame, reward, done, info = env.step(action)
            frame = preprocess_frame(frame)
            buffer.append(frame)
            next_state = copy(buffer)

            agent.remember([
                last_state,
                action,
                reward,
                next_state,
                done])
        else:
            action = random.randint(0, env.action_space.n - 1)
            frame, reward, done, info = env.step(action)
            frame = preprocess_frame(frame)
            buffer.append(frame)


def preprocess_frame(frame):
    frame = cv2.resize(frame, (84, 110))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame[-84:, :]


if __name__ == "__main__":
    env = gym.make("Breakout-v4")
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
                state = copy(buffer)
                action = agent.act(np.array(buffer).reshape(1, 4, 84, 84))
                frame, reward, done, info = env.step(action)
                frame = preprocess_frame(frame)
                buffer.append(frame)
                next_state = copy(buffer)

                if info["ale.lives"] < lives:
                    reward = -5
                    lives = info["ale.lives"]

                total_reward += reward

                agent.remember([state, action, reward, next_state, done])
                agent.replay(iteration_counter)

                if done:
                    print("EPOCH {} END. Iterations: {} Avg. loss: {} Reward: {} EPSILON: {} Memory: {} MB".format(
                        i,
                        iteration_counter,
                        agent.get_avg_loss(),
                        total_reward,
                        agent.EPSILON,
                        round(process.memory_info().rss * 10 ** (-6)), 2))
                    break
                iteration_counter += 1
            agent.epsilon_decay()
            if i%100 == 0:
                agent.model.save("/content/gdrive/My Drive/model2_iter{}.h5".format(i))
    except KeyboardInterrupt:
        pass

    agent.model.save("/content/gdrive/My Drive/model_final.h5")

    plt.plot(agent.loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()