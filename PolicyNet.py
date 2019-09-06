'''
Neural Network model, partially adapted from https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/policy_value_net_tensorflow.py 
and modified to work with Sudoku instead of Gomoku
'''
import Config as cfg

import numpy as np
import tensorflow as tf

class PolicyValueNet():
	def __init__(self, model_file = None):
		# define the tensorflow neural network
		# 1. input: board row * board col * value
		self.input_states = tf.placeholder(tf.float32, shape = [None, cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE])
		# 2. Common Networks Layers
		self.conv1 = tf.layers.conv2d(inputs = self.input_states, filters = 32, kernel_size = [3, 3],
			padding = "same", data_format = "channels_last", activation = tf.nn.relu)
		self.conv2 = tf.layers.conv2d(inputs = self.conv1, filters = 64, kernel_size = [3, 3],
			padding = "same", data_format = "channels_last", activation = tf.nn.relu)
		self.conv3 = tf.layers.conv2d(inputs = self.conv2, filters = 128, kernel_size = [3, 3],
			padding = "same", data_format = "channels_last", activation = tf.nn.relu)
		# 3.1. Action Networks
		self.action_conv = tf.layers.conv2d(inputs = self.conv3, filters = 9, kernel_size = [1, 1], 
			padding = "same", data_format = "channels_last", activation = tf.nn.relu)
		self.action_conv_flat = tf.reshape(self.action_conv, [-1, cfg.TOTAL_ACTIONS])
		# 3.2. Full connected layer, the output is the log probability of moves on each position on the board
		self.action_fc = tf.layers.dense(inputs = self.action_conv_flat, units = cfg.TOTAL_ACTIONS,
			activation = tf.nn.log_softmax)
		# 4. Evaluation Networks
		self.evaluation_conv = tf.layers.conv2d(inputs = self.conv3, filters = 2, kernel_size = [1, 1],
			padding = "same", data_format = "channels_last", activation = tf.nn.relu)
		self.evaluation_conv_flat = tf.reshape(self.evaluation_conv, [-1, 2 * cfg.TOTAL_GRIDS])
		self.evaluation_fc_tmp = tf.layers.dense(inputs = self.evaluation_conv_flat, units = 64, 
			activation = tf.nn.relu)
		self.evaluation_fc = tf.layers.dense(inputs = self.evaluation_fc_tmp, units = 1, 
			activation = tf.nn.tanh)
		# define the loss function
		# 1.1. value loss function
		self.labels = tf.placeholder(tf.float32, shape = [None, 1])
		self.value_loss = tf.losses.mean_squared_error(self.labels, self.evaluation_fc)
		# 1.2. Policy loss function
		self.mcts_probs = tf.placeholder(tf.float32, shape = [None, cfg.TOTAL_ACTIONS])
		self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
		# 1.3. L2 penelty
		l2_penalty_beta = 1e-4
		vars = tf.trainable_variables()
		l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if "bias" not in v.name.lower()])
		# 1.4. Add up to the loss function
		self.loss = self.value_loss + self.policy_loss + l2_penalty

		# Define the optimizer for training
		self.learning_rate = tf.placeholder(tf.float32)
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

		# Define the tensorflow Session
		self.session = tf.Session()

		# Calculate policy entropy
		self.entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

		# Initialization variables
		init = tf.global_variables_initializer()
		self.session.run(init)

		# Model saving and restoring
		self.saver = tf.train.Saver()
		if model_file:
			self.restore_model(model_file)


	def policy_value(self, state_batch):
		# input: a batch of state
		# output: a batch of action probabilities and state values
		log_act_probs, value = self.session.run([self.action_fc, self.evaluation_fc], 
			feed_dict = {self.input_states: state_batch})
		act_probs = np.exp(log_act_probs)
		return act_probs, value

	def policy_val_func(self, board):
		# input: board
		# output: a list of (action, probability) tuples for each available action and the score of the board state
		available_actions = board.get_available_actions()
		cur_state = np.ascontiguousarray(np.reshape(board.get_channel_state(), 
			(-1, cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE)))
		act_probs, value = self.policy_value(cur_state)
		act_probs = zip(available_actions, act_probs[0][available_actions])
		return act_probs, value[0][0]


	def train_step(self, state_batch, mcts_probs, result_batch, lr):
		result_batch = np.reshape(result_batch, (-1, 1))
		loss, entropy, _ = self.session.run([self.loss, self.entropy, self.optimizer], 
			feed_dict = {self.input_states: state_batch, self.mcts_probs: mcts_probs, 
			self.labels: result_batch, self.learning_rate: lr})
		return loss, entropy


	def save_model(self, model_path):
		self.saver.save(self.session, model_path)


	def restore_model(self, model_path):
		self.saver.restore(self.session, model_path)