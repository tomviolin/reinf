#!/usr/bin/env python3
import gymnasium # as gym
import numpy as np
from numpy import exp
from queue import Queue
import threading
# Ensure you have the required packages installed:
env = gymnasium.make("InvertedDoublePendulum-v4", render_mode='rgb_array')  # Use 'rgb_array' for rendering
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
rgb = [env.render()]  # Initial render to get the first frame
killflag = False  # Flag to indicate if the program should exit
q = Queue()  # Create a queue to hold the data for plotting
#imageq = Queue()  # Create a queue to hold the data for plotting

def killIn(secs):

    """Function to set the kill flag after a certain number of seconds."""
    import time
    global killflag
    time.sleep(secs)
    killflag = True  # Set the kill flag to True after the specified time
    print("Kill flag set to True after {} seconds".format(secs))

def DisplayThread():
    global killflag
    import cv2
    import matplotlib
    from matplotlib import pyplot as plt
    """Function to plot the collected data in a separate thread."""
    matplotlib.use('agg')  # Use TkAgg backend for matplotlib
    while not killflag:
        print("DISPTHREAD: Waiting for episode number...")
        [maxepno, qrgb] = q.get()
        if maxepno<0:
            if cv2.waitKey(1) & 0xFF in set([ord('q'), 27]):  # Exit if 'q' is pressed
                cv2.destroyAllWindows()  # Close all OpenCV windows
                cv2.waitKey(1000)
                threading.Thread(target=killIn, args=(1,)).start()  # Start a thread to set the kill flag after 1 second
                return  # Exit the thread if the signal is received
            else:
                continue
        fig=plt.figure(figsize=(12, 4.5)) # get a fig handle so we can get the canvas RGBA buffer later
        plt.clf()  # Clear the current figure
        epnoarray = np.array(epno[:maxepno])  # Get the episode numbers up to the max episode number
        epnoarray = epnoarray[-1] - epnoarray + 1  # Normalize episode numbers to start from 1
        epnoarray = epnoarray / np.max(epnoarray) 
        #epnoarray = 100.0/(100.0 + epnoarray)  # Invert the episode numbers to compress the x-axis
        epnoarray = exp(-(epnoarray)) # Exponential decay to compress the x-axis

        plt.xticks(epnoarray[:maxepno:1000], epno[:maxepno:1000], rotation=80)
        plt.plot(epnoarray,q10s[:maxepno],label='0.10 quantile',lw=0.05) # Plot the collected data
        plt.plot(epnoarray,q50s[:maxepno],label='0.50 quantile',lw=0.05) # Plot the collected data
        plt.plot(epnoarray,avgs[:maxepno],label='Average',lw=0.05)
        plt.plot(epnoarray,q90s[:maxepno],label='0.90 quantile',lw=0.05) # Plot the collected data
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

        fig=plt.figure(figsize=(12, 4.5)) # get a fig handle so we can get the canvas RGBA buffer later
        plt.clf()  # Clear the current figure
        plt.xticks(epnoarray[::1000], epno[:maxepno:1000], rotation=80)
        plt.plot(epnoarray,eplen[:maxepno],label='Episode Length', lw=0.1) # Plot the collected data
        # take the last 1000 episodes into their own arrays
        epnoarray_last1000 = epnoarray[:]
        eplen_last1000 = np.array(eplen[:maxepno][:])

        filt = (eplen_last1000 > np.quantile(eplen_last1000, 0.90))

        print(filt) 
        print(filt.shape)
        print(filt.dtype)
        x = epnoarray_last1000[filt]
        print(eplen_last1000.dtype)
        y = eplen_last1000[filt]
        plt.plot(epnoarray_last1000[filt], eplen_last1000[filt], 'ro', markersize=1, label='Filtered Episode Lengths')  # Plot filtered episode lengths

        # add dotted zero line
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.title("Length of Episodes in InvertedPendulumv5")
        plt.xlabel("Episode Number")
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

        for i in range(0, len(qrgb)):
            qrgbi = 1.0*qrgb[i]
            xlen = qrgbi.shape[1]

            qrgbi[:,:xlen//2,:] = qrgbi[:,:xlen//2,::-1]  # Flip the left half of the image horizontally
            qrgbi -= qrgbi.min()
            qrgbi /= qrgbi.max()
            """
             1                           *
                                *
                         *
                    *
                 *
                *
                ---------------------------
             0  0                         1
            """
            # apply a sigmoid function to compress the values between 0 and 1
            qrgbi = 1.0 / (1.0 + np.exp(-10 * (qrgbi - 0.5)))  # Sigmoid function to compress values
            cv2.imshow("Swimmer-v5", qrgbi) # Show the last rendered frames
            if cv2.waitKey(5) & 0xFF in set([ord('q'), 27]):  # Exit if 'q' is pressed
                cv2.destroyAllWindows()
                cv2.waitKey(1000)  # Wait for a key press to ensure the window closes properly
                threading.Thread(target=killIn, args=(1,)).start()  # Start a thread to set the kill flag after 1 second
                return
        if cv2.waitKey(1) & 0xFF in set([ord('q'), 27]):  # Exit if 'q' is pressed
            cv2.destroyAllWindows()
            cv2.waitKey(1000)
            threading.Thread(target=killIn, args=(1,)).start()
            return  # Exit the thread if the signal is received
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


while not killflag:

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
        #print(f"Episode {nepisodes} finished after {i + 1} timesteps")
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
        # Now plot the collected data, but only every 1000 episodes
        if nepisodes > 1 and nepisodes % 1000 == 1:
            rgb.append(env.render())
            q.put_nowait([nepisodes,rgb+[]])  # Put the episode number and the last rendered frame in the queue
            print("just put episode number and rgb in queue")
            rgb = []
            # Start a new thread to display the plot
        observation, info = env.reset()
        i = 0
        plotstuff=[]
        if not killflag and nepisodes % 300 == 0:
            q.put_nowait([-1,[]])  # Put a signal to the display thread to check the keyboard and exit if user presses q or esc
            print("just put -1 in queue to check for kill signal")
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

