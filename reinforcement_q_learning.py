#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement Learning (DQN) Tutorial
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_
            `Mark Towers <https://github.com/pseudo-rnd-thoughts>`_


This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent
on the CartPole-v1 task from `Gymnasium <https://gymnasium.farama.org>`__.

You might find it helpful to read the original `Deep Q Learning (DQN) <https://arxiv.org/abs/1312.5602>`__ paper

**Task**

The agent has to decide between two actions - moving the cart left or
right - so that the pole attached to it stays upright. You can find more
information about the environment and other more challenging environments at
`Gymnasium's website <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`__.

.. figure:: /_static/img/doubleinv5.gif
   :alt: CartPole

   CartPole

As the agent observes the current state of the environment and chooses
an action, the environment *transitions* to a new state, and also
returns a reward that indicates the consequences of the action. In this
task, rewards are +1 for every incremental timestep and the environment
terminates if the pole falls over too far or the cart moves more than 2.4
units away from center. This means better performing scenarios will run
for longer duration, accumulating larger return.

The CartPole task is designed so that the inputs to the agent are 4 real
values representing the environment state (position, velocity, etc.).
We take these 4 inputs without any scaling and pass them through a 
small fully-connected network with 2 outputs, one for each action. 
The network is trained to predict the expected value for each action, 
given the input state. The action with the highest expected value is 
then chosen.


**Packages**


First, let's import needed packages. Firstly, we need
`gymnasium <https://gymnasium.farama.org/>`__ for the environment,
installed by using `pip`. This is a fork of the original OpenAI
Gym project and maintained by the same team since Gym v0.19.
If you are running this in Google Colab, run:

.. code-block:: bash

   %%bash
   pip3 install gymnasium[classic_control]

We'll also use the following from PyTorch:

-  neural networks (``torch.nn``)
-  optimization (``torch.optim``)
-  automatic differentiation (``torch.autograd``)

"""

import gymnasium as gym
import math,os,sys,time
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys, os

"""
import inverted_double_pend
gym.envs.register(
    id='InvertedDoublePend',
    entry_point='inverted_double_pend:InvertedDoublePendEnv',
    max_episode_steps=1000,
    reward_threshold=500.0,
)
#default to training mode with no rendering
"""
env_render = 'human'
training = True
readq = False


def mypause(interval):
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw()
            time.sleep(interval)
        else:
            time.sleep(interval)

    else:
        time.sleep(interval)



if len(sys.argv) > 1:
    # "test" = test the model
    if "test" in sys.argv[1:]:
        #print("Testing the model...")
        # if testing, we don't need to train, just run the test_model function
        env_render = 'human'
        training = False
    if "render" in sys.argv[1:]:
        #print("Rendering the environment...")
        # if rendering, we want to see the environment
        env_render = 'human'
    if "nogui" in sys.argv[1:]:
        #print("Running without GUI...")
        # if nogui, we don't want to render the environment
        env_render = None
    if "readq" in sys.argv[1:]:
        readq = True
env = gym.make("Swimmer-v5", render_mode=env_render)
print(env.action_space)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classes:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save_dict(self, path):
        """Save the memory to a file."""
        torch.save(self.memory, path)

    def load_dict(self, path):
        pass

        """Load the memory from a file.
        self.memory = torch.load(path, map_location=device)
        if not isinstance(self.memory, deque):
            raise ValueError("Loaded memory is not a deque.")
        self.memory = deque(self.memory, maxlen=self.memory.maxlen)
        """

######################################################################
# Now, let's define our model. But first, let's quickly recap what a DQN is.
#
# DQN algorithm
# -------------
#
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
#
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# :math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
# :math:`R_{t_0}` is also known as the *return*. The discount,
# :math:`\gamma`, should be a constant between :math:`0` and :math:`1`
# that ensures the sum converges. A lower :math:`\gamma` makes 
# rewards from the uncertain far future less important for our agent 
# than the ones in the near future that it can be fairly confident 
# about. It also encourages agents to collect reward closer in time 
# than equivalent rewards that are temporally far away in the future.
#
# The main idea behind Q-learning is that if we had a function
# :math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
#
# .. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)
#
# However, we don't know everything about the world, so we don't have
# access to :math:`Q^*`. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# :math:`Q^*`.
#
# For our training update rule, we'll use a fact that every :math:`Q`
# function for some policy obeys the Bellman equation:
#
# .. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
#
# The difference between the two sides of the equality is known as the
# temporal difference error, :math:`\delta`:
#
# .. math:: \delta = Q(s, a) - (r + \gamma \max_a' Q(s', a))
#
# To minimize this error, we will use the `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of :math:`Q` are very noisy. We calculate
# this over a batch of transitions, :math:`B`, sampled from the replay
# memory:
#
# .. math::
#
#    \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)
#
# .. math::
#
#    \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}
#
# Q-network
# ^^^^^^^^^
#
# Our model will be a feed forward  neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing :math:`Q(s, \mathrm{left})` and
# :math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
# network). In effect, the network is trying to predict the *expected return* of
# taking each action given the current input.
#

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128).to(device)
        self.layer3 = nn.Linear(128, 64).to(device)
        self.layer4 = nn.Linear(64,32  ).to(device)
        self.layer5= nn.Linear(32,  16).to(device)
        self.layer6= nn.Linear(16,   8).to(device)
        self.layerfinal = nn.Linear(8  , n_actions).to(device)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        print(f"GPU device: {device}, x: {x.device}")
        x = F.relu(self.layer1(x)).to(device)
        x = F.relu(self.layer3(x)).to(device)
        x = F.relu(self.layer4(x)).to(device)
        x = F.relu(self.layer5(x)).to(device)
        x = F.relu(self.layer6(x)).to(device)
        return self.layerfinal(x).to(device)

    def save_dict(self, path):
        """Save the model state dict to a file."""
        torch.save(self.state_dict(), path)

    def load_dict(self, path):
        """Load the model state dict from a file."""
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.eval()
    
    def save_model(self, path):
        """Save the model to a file."""
        torch.save(self, path, weights_only=True)

    def load_model(self, path):
        """Load the model from a file."""
        self.__init__()
        self.load_dict(path)
        self.eval()

######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action according to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the duration of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.10
EPS_DECAY = 1000
TAU = 0.05
LR = 4e-4

# Get number of actions from gym action space
n_actions = 1
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
# load a previous save of the model if available
try:
    if readq:
        policy_net.load_dict("doubleinv5_policy_net.pth")
except FileNotFoundError:
    pass
#print("No saved model found, starting from scratch.")
# Create a target network with the same architecture and weights as the policy network
target_net = DQN(n_observations, n_actions).to(device)
# Initialize the target network with the same weights as the policy network
# This is done to stabilize training by using a separate network for
# computing the target Q-values

try:
    if readq:
        target_net.load_dict("doubleinv5_target_net.pth")
except FileNotFoundError:
    #print("No saved model found, starting from scratch.")
    target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# Create a replay memory to store transitions

memory = ReplayMemory(10000)
if readq and os.path.exists("doubleinv5_replay_memory.pth"):
    #print("Loading replay memory from file...")
    memory.load_dict("doubleinv5_replay_memory.pth")


steps_done = 0


def select_action(state):
    global steps_done
    #print(f"select_action: state: {state}, steps_done: {steps_done}")
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if steps_done < 10:
        # for the first 10 steps, we always take a random action
        eps_threshold = 1.0
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        sample_action = env.action_space.sample()
        #if type(sample_action) is np.ndarray:
        #    sample_action = [sample_action[0]]
        #print(f"Taking random action: {sample_action} (eps_threshold: {eps_threshold})")
        #print(f"env.action_space.sample(): {sample_action}")
        return torch.tensor([sample_action], device=device, dtype=torch.float32)

episode_durations = []

user_terminated = False
def on_press(event):
    global user_terminated
    if event.key=='q':
        # if 'q' is pressed, stop the training
        print("Training stopped by user.")
        plt.close('all')
        user_terminated = True

def plot_durations(show_result=False):
    fig = plt.figure(1)
    fig.canvas.mpl_connect('key_press_event', on_press)

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.show(block=False)
    plt.pause(0.01)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    # save the model after every episode
    policy_net.save_dict("doubleinv5_policy_net.pth")
    # save the target network as well
    target_net.save_dict("doubleinv5_target_net.pth")
    # save the replay memory to a file
    memory.save_dict("doubleinv5_replay_memory.pth")

######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By definition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network is updated at every step with a 
# `soft update <https://arxiv.org/pdf/1509.02971.pdf>`__ controlled by 
# the hyperparameter ``TAU``, which was previously defined.
#

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device).unsqueeze(1)
    action_batch = action_batch.type(torch.int64)  # Ensure action_batch is of type int64
    reward_batch = torch.cat(batch.reward).to(device)
    #print(f"optimize_model: state_batch: {state_batch}, action_batch: {action_batch}, reward_batch: {reward_batch}, non_final_next_states: {non_final_next_states}")
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    print(f"state_batch: {state_batch.shape}, action_batch: {action_batch.shape}, reward_batch: {reward_batch.shape}, non_final_next_states: {non_final_next_states.shape}")
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    torch.device(device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


######################################################################

# Below, you can find the main training loop. At the beginning we reset
# the environment and obtain the initial ``state`` Tensor. Then, we sample
# an action, execute it, observe the next state and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set to 600 if a GPU is available, otherwise 50 
# episodes are scheduled so training does not take too long. However, 50 
# episodes is insufficient for to observe good performance on CartPole.
# You should see the model constantly achieve 500 steps within 600 training 
# episodes. Training RL agents can be a noisy process, so restarting training
# can produce better results if convergence is not observed.
#

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 60000
else:
    num_episodes = 50


# if you want to resume training from a previous checkpoint, uncomment the line below
# num_episodes = 0
episode_durations = []


# testing the model
def test_model(num_episodes=10):
    for _ in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        for t in count():
            action = policy_net(state).flatten() #.max(1).indices.view(1, 1)
            observation, reward, terminated, truncated, _ = env.step(action.cpu().detach().numpy())
            total_reward += reward
            if terminated or truncated:
                #print(f"Episode finished after {t + 1} timesteps with total reward {total_reward}")
                break
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    env.close()



if (not training):
    # if we are testing the model or rendering the environment, we don't need to train
    #print("Testing the model...")
    test_model(num_episodes)
    env.close()
    sys.exit(0)

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        #print(f"type(state): {type(state)}, state: {state}, t: {t}, episode: {i_episode}")
        action = select_action(state).flatten()
        # action = action.cpu()
        #print(f"action: {action}, type(action): {type(action)} action.shape: {action.shape}")
        observation, reward, terminated, truncated, _ = env.step(action())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            if len(episode_durations) % 100 == 0:
                plot_durations()
            break
    if user_terminated:
        print("Training terminated by user.")
        break
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

######################################################################
# Here is the diagram that illustrates the overall resulting data flow.
#
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
#
# Actions are chosen either randomly or based on a policy, getting the next
# step sample from the gym environment. We record the results in the
# replay memory and also run optimization step on every iteration.
# Optimization picks a random batch from the replay memory to do training of the
# new policy. The "older" target_net is also used in optimization to compute the
# expected Q values. A soft update of its weights are performed at every step.
#
