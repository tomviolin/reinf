#!/usr/bin/env python3
import gymnasium as gym
from PIL import Image
import cv2
env = gym.make("CartPole-v1", render_mode='rgb_array')  # Use 'rgb_array' for rendering
observation, info = env.reset()
cv2.namedWindow("CartPole", cv2.WINDOW_NORMAL)  # Create a resizable window
i = 0
while True:

    if False:  # Change this to True if you want to render every step
        rgb = env.render()  # Render the environment every 100 steps
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
        cv2.imshow("CartPole", rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        rgb = env.render()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("CartPole", rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break
        print("Episode terminated or truncated")

        print("Episode finished after {} timesteps".format(i + 1))
        observation, info = env.reset()
        i = 0
    # Uncomment the next line to see the observation and info
    #print(i, observation, info)
    #if i % 10 == 0:
        #print(f"Step {i}: Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        # Import necessary libraries
        # You can add a sleep here if you want to slow down the rendering
        # import time
        # time.sleep(0.1)  # Adjust the sleep time as needed
    i+=1

env.render()  # Final render to show the last state
# Close the environment
env.close()



"""
episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
"""

