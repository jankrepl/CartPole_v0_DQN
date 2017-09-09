# CartPole_v0_DQN
TensorFlow implementation of a Deep Q Network (**DQN**) solving the problem of balancing a pole on cart.
Environment provided by the OpenAI gym.

## Code

### Running
```
python Main.py
```

### Dependencies
*  collections
*  gym
*  numpy
*  random
*  tensorflow

## Detailed Description
### Problem Statement and Environment
The goal is to move the cart to the left and right in a way that the pole on top of it does not fall down. The states 
of the environment are composed of 4 elements - **cart position** (x), **cart speed** (xdot),
**pole angle** (theta) and **pole angular velocity** (thetadot). For each time step when the pole is still on the cart
we get a reward of 1. The problem is considered to be solved if for 100 consecutive
episodes the average reward is at least 195.


If we translate this problem into reinforcement learning terminology:
* action space is **0** (left) and **1** (right)
* state space is a set of all 4-element lists with all possible combinations of values of x, xdot, theta, thetadot

---
### DQN
Deep Q Network combines reinforcement learning with deep learning. A huge advantage of DQN over tabular methods is that
we do not have to discretize the state space. Instead, we train a neural network as a function approximation of the **action
value function Q(s,a)**. This approach enables us to generalize and assign action values to states-action pairs we 
have never visited before. Below we describe important ideas behind the algorithm and implementation.


#### Target
In supervised learning, we are given true labels/targets on which we can train the model. However, in the case
of reinforcement learning, there is no supervisor that provides us with correct Q(s,a) values. However, the trick
is to use the Q backtrack to "guess" the true value/label/target and use it in training of the neural network. The below
picture summarizes this idea:

![target](https://user-images.githubusercontent.com/18519371/30241140-cf9be86e-957d-11e7-939e-0c6ad377e5ca.png)

#### Memory
As suggested in the now notorious [paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) from DeepMind we use
a concept of **memory**. It is a database of the latest states, actions and rewards. More specifically, at each time step 
5 pieces of information are stored into the memory:
* **s_old** - the state before taking an action
* **a** - action
* **r** - reward obtained after taking an action a in the state s_old
* **s_new** - the state after taking an action a
* **d** - boolean representing whether the pole fell or not


#### Training and replaying the memory
We train our network at each timestep but instead of taking the most recent observations into account we **randomly 
sample a batch from our memory**. This approach guarantees that we do not learn from highly correlated
sequential observations but instead from (ideally) independent ones.



## Resources and links
### Amazing Keras Implementations and tutorials
* https://keon.io/deep-q-learning/
* https://threads-iiith.quora.com/Deep-Q-Learning-with-Neural-Networks-on-Cart-Pole
* https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
* https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/

### Official OpenAI baseline
* https://github.com/openai/baselines/tree/master/baselines/deepq


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
