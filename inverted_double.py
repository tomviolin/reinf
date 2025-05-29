#!/usr/bin/env python3
import gymnasium # as gym
import numpy as np
from numpy import exp
from queue import Queue
import threading
# Ensure you have the required packages installed:
env = gymnasium.make("InvertedDoublePendulum-v5", render_mode='rgb_array')  # Use 'rgb_array' for rendering
observation, info = env.reset()
#cv2.namedWindow("Swimmer-v5", cv2.WINDOW_NORMAL)  # Create a resizable window
i = 0
nepisodes = 0
plotstuff=[]
avgs = []
q50s = []
q10s = []
q90s = []
epno = []
eplen = []

q = Queue()  # Create a queue to hold the data for plotting


def DisplayThread():
    import cv2
    import matplotlib
    from matplotlib import pyplot as plt
    """Function to plot the collected data in a separate thread."""
    matplotlib.use('agg')  # Use TkAgg backend for matplotlib
    while True:
        print("DISPTHREAD: Waiting for episode number...")
        maxepno = q.get()
        rgb = env.render()
        fig=plt.figure(figsize=(15, 4.5)) # get a fig handle so we can get the canvas RGBA buffer later
        plt.clf()  # Clear the current figure
        epnoarray = np.array(epno[:maxepno])  # Get the episode numbers up to the max episode number
        epnoarray = epnoarray[-1] - epnoarray + 1  # Normalize episode numbers to start from 1
        epnoarray = epnoarray / np.max(epnoarray) 
        #epnoarray = 100.0/(100.0 + epnoarray)  # Invert the episode numbers to compress the x-axis
        epnoarray = exp(-(epnoarray)) # Exponential decay to compress the x-axis

        plt.xticks(epnoarray[:maxepno:1000], epno[:maxepno:1000], rotation=80)
        plt.plot(epnoarray,q10s[:maxepno],label='0.10 quantile') # Plot the collected data
        plt.plot(epnoarray,q50s[:maxepno],label='0.50 quantile') # Plot the collected data
        plt.plot(epnoarray,avgs[:maxepno],label='Average')
        plt.plot(epnoarray,q90s[:maxepno],label='0.90 quantile') # Plot the collected data
        # add dotted zero line
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.title("InvertedPendulumv5 Observation Plot")
        plt.xlabel("Episode Number")
        plt.ylabel("Reward Values")
        plt.grid(True)  # Add a grid for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.legend()  # Add a legend to the plot
        fig.canvas.draw() # Draw the canvas to get the updated image
        #we want to use opencv imshow to show the plot
        image_data = np.frombuffer(fig.canvas.renderer.tostring_argb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert to BGR format
        image_data = image_data[:, :, [3, 2, 1, 0]]  # Convert ARGB to BGRA

        plt.close('all')

        fig=plt.figure(figsize=(15, 4.5)) # get a fig handle so we can get the canvas RGBA buffer later
        plt.clf()  # Clear the current figure
        plt.xticks(epnoarray[::1000], epno[:maxepno:1000], rotation=80)
        plt.plot(epnoarray,eplen[:maxepno],label='Episode Length') # Plot the collected data
        # add dotted zero line
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.title("Length of Episodes in InvertedPendulumv5")
        plt.xlabel("Episode Number", rotation=80)
        plt.ylabel("Episode Length")
        plt.grid(True)  # Add a grid for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        fig.canvas.draw() # Draw the canvas to get the updated image
        #we want to use opencv imshow to show the plot
        image_data2 = np.frombuffer(fig.canvas.renderer.tostring_argb(), dtype=np.uint8)
        image_data2 = image_data2.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert to BGR format
        image_data2 = image_data2[:, :, [3, 2, 1, 0]]  # Convert ARGB to BGRA

        plt.close('all')


        #pltrgb = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGR)
        cv2.imshow("Swimmer-v5 Plot", image_data)  # Show the plot using OpenCV
        cv2.imshow("Episode Length Plot", image_data2)  # Show the plot using OpenCV
        cv2.imshow("Swimmer-v5", rgb) # Show the last rendered frame
        if cv2.waitKey(1) & 0xFF in set([ord('q'), 27]):  # Exit if 'q' is pressed
            return
        print("Episode terminated or truncated")

        print("Episode finished after {} timesteps".format(i + 1))
        print(i)
        print(f"observation={observation}")
        print(f"reward={reward}")
        print(f"terminated={terminated}")
        print(f"truncated={truncated}") 
        print(f"info={info}")


# Start the display thread to plot the data
# it will serve as a worker thread mostly waiting for 
# the queue to send a request.  The request is simply
# the episode number. The plotted data will update the
# plots to be current as of the episode number passed.
# so this thread will only read the data, and not modify it.
# since threads share the same memory space, we can read
# everything from the global data, but we do not need to
# declare it global in the function because we will only be 
# reading it, not modifying it.

dispThread = threading.Thread(target=DisplayThread, daemon=True)
dispThread.start()  # Start the display thread


while True:

    """if i % 3000 == 0:  # Change this to whatever frequency you want to render
        rgb = env.render()  # Render the environment every 100 steps
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
        cv2.imshow("Swimmer-v5", rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break
    """
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    plotstuff.append(reward)  # Collect rewards for plotting
    if terminated or truncated:
        nepisodes += 1
        print(f"Episode {nepisodes} finished after {i + 1} timesteps")
        # Calculate quantiles and averages for the last episode rewards every time an episode ends
        q50s.append(np.quantile((plotstuff),.50))  # Calculate the 0.50 quantile of the last episode rewards
        q10s.append(np.quantile((plotstuff),.10))  # Calculate the 0.10 quantile of the last episode rewards
        q90s.append(np.quantile((plotstuff),.90))  # Calculate the 0.90 quantile of the last episode rewards
        avgs.append(np.mean(plotstuff))            # Calculate the average of the last episode rewards
        epno.append(nepisodes)  # Store the episode number
        eplen.append(i + 1)
        """if len(q50s) > 500:
            q50s = q50s[-500:]  # Keep only the last 1000 values
            q10s = q10s[-500:]
            q90s = q90s[-500:]
            avgs = avgs[-500:]
            epno = epno[-500:]
        """
        # Now plot the collected data, but only every 100 episodes
        if nepisodes % 1000 == 0:
            q.put_nowait(nepisodes)
            # Start a new thread to display the plot
        observation, info = env.reset()
        i = 0
        plotstuff=[]
        
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

