from keras.models import load_model
from collections import deque
import cv2
import gym
import numpy as np
import time

def preprocess_frame(frame):
    frame = cv2.resize(frame, (84, 110))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame[-84:, :]

buffer = deque(maxlen=4)
model = load_model("model2_iter400.h5")
env = gym.make("SpaceInvadersDeterministic-v4")
env.reset()

while True:
    if len(buffer) == 4:
        x = np.array(buffer).reshape(1,4,84,84)
        prediction = model.predict(x)
        action = np.argmax(prediction)
    else:
        action = 0
    
    frame, reward, done, info = env.step(action)
    frame = preprocess_frame(frame)
    buffer.append(frame)
    env.render()
    time.sleep(0.01)
    if done:
        env.reset()
        buffer = deque(maxlen=4)

env.close()
