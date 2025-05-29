#!/usr/bin/env python3
import gymnasium # as gym
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Ensure you have the required packages installed:
env = gymnasium.make("Swimmer-v5", render_mode='rgb_array')  # Use 'rgb_array' for rendering
observation, info = env.reset()
cv2.namedWindow("Swimmer-v5", cv2.WINDOW_NORMAL)  # Create a resizable window
i = 0

plotstuff=[]
avgs = []
q50s = []
q10s = []
q90s = []
while True:

    if i % 3000 == 0:  # Change this to whatever frequency you want to render
        rgb = env.render()  # Render the environment every 100 steps
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
        cv2.imshow("Swimmer-v5", rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    plotstuff.append(reward)  # Collect rewards for plotting
    if terminated or truncated:
        rgb = env.render()
        fig=plt.figure(figsize=(5, 2.5)) # get a fig handle so we can get the canvas RGBA buffer later
        plt.clf()  # Clear the current figure
        q50s.append(np.quantile((plotstuff),.50))  # Calculate the 0.50 quantile of the last episode rewards
        q10s.append(np.quantile((plotstuff),.10))  # Calculate the 0.10 quantile of the last episode rewards
        q90s.append(np.quantile((plotstuff),.90))  # Calculate the 0.90 quantile of the last episode rewards
        avgs.append(np.mean(plotstuff))            # Calculate the average of the last episode rewards
        plt.plot(q10s,label='0.10 quantile') # Plot the collected data
        plt.plot(q50s,label='0.50 quantile') # Plot the collected data
        plt.plot(avgs,label='Average')
        plt.plot(q90s,label='0.90 quantile') # Plot the collected data
        # add dotted zero line
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.title("Swimmer-v5 Observation Plot")
        plt.xlabel("Time Step")
        plt.ylabel("Observation Values")
        plt.grid(True)  # Add a grid for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.legend()  # Add a legend to the plot
        fig.canvas.draw() # Draw the canvas to get the updated image
        #we want to use opencv imshow to show the plot
        image_data = np.frombuffer(fig.canvas.renderer.tostring_argb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        print(image_data.shape)
        # Convert to BGR format
        image_data = image_data[:, :, [3, 2, 1, 0]]  # Convert ARGB to BGRA
        #image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGRA)


        #pltrgb = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGR)
        cv2.imshow("Swimmer-v5 Plot", image_data)  # Show the plot using OpenCV
        cv2.imshow("Swimmer-v5", rgb) # Show the last rendered frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break
        print("Episode terminated or truncated")

        print("Episode finished after {} timesteps".format(i + 1))
        print(i)
        print(f"observation={observation}")
        print(f"reward={reward}")
        print(f"terminated={terminated}")
        print(f"truncated={truncated}") 
        print(f"info={info}")
        observation, info = env.reset()
        i = 0
        plotstuff=[]
        
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

