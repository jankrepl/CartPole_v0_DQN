import random
from collections import deque

import numpy as np
import tensorflow as tf


class Neural_Network:
    def __init__(self, parameter_dict):
        # Input attributes
        self.MEMORY_SIZE = parameter_dict['MEMORY_SIZE']
        self.BATCH_SIZE = parameter_dict['BATCH_SIZE']
        self.HL_1_SIZE = parameter_dict['HL_1_SIZE']
        self.HL_2_SIZE = parameter_dict['HL_2_SIZE']
        self.GAMMA = parameter_dict['GAMMA']

        # Initialize variables
        self.memory = deque()  # popleft-> [element_oldest,...., element_newest] <- append
        self.timestep = 0  # this counts the number of times we trained (or tried to train) our network

        # LAST RESULT TRACKER
        self.most_recent_record = tf.Variable(0, dtype=tf.int32)  # it will log the number of timesteps we held the pole for in previous try
        tf.summary.scalar('most_recent_record', self.most_recent_record)

        self.__design_network()
        self.__initialize_network()

        # Summaries
        global summary_writer
        summary_writer = tf.summary.FileWriter('/Users/jankrepl/Desktop/TFsummaries', graph=self.session.graph)

    def __design_network(self):
        """Designing the network

        """
        # INFERENCE - FORWARD PASS
        self.input_states = tf.placeholder(tf.float32, [None, 4], name='Input_States')  # None enables variable row size
        # which is great for us, because in training we use it in batches and in choose action we use it alone
        self.input_actions = tf.placeholder(tf.float32, [None, 2],
                                            name='Input_Action_OHR')  # None enables variable row size
        self.output = tf.placeholder(tf.float32, [None], name='Q_prediction')  # None enables variable row size

        self.W1 = tf.Variable(tf.truncated_normal([4, self.HL_1_SIZE]), name='W1')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[self.HL_1_SIZE]), name='b1')
        self.HL_1 = tf.nn.relu(tf.matmul(self.input_states, self.W1) + self.b1, name='HL1')
        self.W2 = tf.Variable(tf.truncated_normal([self.HL_1_SIZE, self.HL_2_SIZE]), name='W2')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[self.HL_2_SIZE]), name='b2')
        self.HL_2 = tf.nn.relu(tf.matmul(self.HL_1, self.W2) + self.b2, name='HL2')
        self.W3 = tf.Variable(tf.truncated_normal([self.HL_2_SIZE, 2]), name='W3')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[2]), name='b3')
        self.Q_ohr = tf.nn.relu(tf.matmul(self.HL_2, self.W3) + self.b3, name='Q_ohr')

        self.Q = tf.reduce_sum(tf.multiply(self.Q_ohr, self.input_actions), reduction_indices=1, name='Q')

        # LOSS
        self.loss = tf.reduce_mean(tf.square(self.output - self.Q), name='loss')
        tf.summary.scalar("loss", self.loss)

        self.optimizer = tf.train.AdamOptimizer(0.0001)
        self.train_op = self.optimizer.minimize(self.loss)

        # Summaries merge
        global merged_summary_op
        merged_summary_op = tf.summary.merge_all()

    def __initialize_network(self):
        # Initialize new session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def memorize(self, s_old, action, reward, s_new, done):
        """ Inserts the most recent SARS and done into the memory - a is saved in the one hot representation

        :param s_old: old state
        :type s_old: ndarray
        :param action: 0 or 1
        :type action: int
        :param reward: reward
        :type reward: int
        :param s_new: new state
        :type s_new: ndarray
        :param done: is finished
        :type done: bool
        """
        # Convert action to one hot representation
        a_ohr = np.zeros(2)
        a_ohr[action] = 1

        # Make sure they have the right dimensions
        s_old.shape = (4,)
        a_ohr.shape = (2,)
        s_new.shape = (4,)

        # Add into memory and if necessary pop oldest memory
        memory_element = tuple((s_old, a_ohr, reward, s_new, done))
        self.memory.append(memory_element)
        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.popleft()

    def train(self):
        """ Samples a minibatch from the memory and based on it trains the network

        """
        self.timestep += 1

        # Just make sure that it breaks at the beginning when memory is not big enough < batch_size
        if len(self.memory) < self.BATCH_SIZE:
            print('The memory is too small to train')
            return

        # Sample from memory

        mini_batch = random.sample(self.memory, self.BATCH_SIZE)  # sampling without replacement
        batch_s_old = [element[0] for element in mini_batch]
        batch_a = [element[1] for element in mini_batch]
        batch_r = [element[2] for element in mini_batch]
        batch_s_new = [element[3] for element in mini_batch]
        batch_d = [element[4] for element in mini_batch]

        # We need to generate labels = targets for each element of the minibatch
        # evaluate Q for all s_new
        Q_new = self.Q_ohr.eval(feed_dict={self.input_states: batch_s_new})

        # Generate targets
        batch_y = []
        for i in range(self.BATCH_SIZE):
            if batch_d[i]:
                batch_y.append(batch_r[i])
            else:
                batch_y.append(batch_r[i] + self.GAMMA * max(Q_new[i]))

        # Train network based on your subsample of memory
        self.train_op.run(feed_dict={
            self.output: batch_y,
            self.input_states: batch_s_old,
            self.input_actions: batch_a,
        })

        # CREATE SUMMARIES

        summary_str = self.session.run(merged_summary_op, feed_dict={
            self.output: batch_y,
            self.input_states: batch_s_old,
            self.input_actions: batch_a,

        })

        summary_writer.add_summary(summary_str, self.timestep)

    def choose_action(self, s_old, epsilon):
        """ Epsilon greedy policy

        :param s_old: old observation
        :type s_old: ndarray
        :param epsilon: greedy algorithm
        :type epsilon: float
        :return: 0 or 1
        :rtype: int
        """
        # just a forward pass and max
        if np.random.rand() < epsilon:
            # Explore
            return np.random.choice([0, 1], 1)[0]
        else:
            s_old.shape = (1, 4)  # make sure it matches the placeholder shape (None, 4)
            return np.argmax(self.Q_ohr.eval(feed_dict={self.input_states: s_old}))
        pass

    def update_MRR(self, value):
        """Updates the most recent record

        :param value: The number of steps that the pole managed to stay on top of the cart
        :type value: int
        :return:
        :rtype:
        """
        assign_op = self.most_recent_record.assign(value)
        self.session.run(assign_op)
