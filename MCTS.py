'''
MCTS tree search algorithm modified from https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_pure.py
'''
import Config as cfg
import MCTSPlay

import numpy as np
import copy

import multiprocessing as mp
import time

def softmax(x):
	probs = np.exp(x - np.max(x))
	probs /= np.sum(probs)
	return probs


class TreeNode(object):
	def __init__(self, parent, prior_p):
		self.parent = parent
		self.children = {}
		self.n_visits = 0
		self.Q = 0
		self.u = 0
		self.P = prior_p


	def expand(self, action_priors):
		for action, prob in action_priors:
			if action not in self.children:
				self.children[action] = TreeNode(self, prob)


	def select(self, c_puct):
		return max(self.children.items(), key = lambda act_node: act_node[1].get_value(c_puct))


	def update(self, leaf_value):
		self.n_visits += 1
		self.Q += 1.0*(leaf_value - self.Q) / self.n_visits


	def update_recursive(self, leaf_value):
		if self.parent:
			self.parent.update_recursive(leaf_value)
		self.update(leaf_value)


	def get_value(self, c_puct):
		self.u = (c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits))
		return self.Q + self.u


	def is_leaf(self):
		return self.children == {}


	def is_root(self):
		return self.parent is None


class MCTS(object):
	def __init__(self, policy_value_fn, c_puct = 5, n_playout = 1000):
		self.root = TreeNode(None, 1.0)
		self.policy = policy_value_fn
		self.c_puct = c_puct
		self.n_playout = n_playout


	def playout(self, board):
		node = self.root
		while True:
			if node.is_leaf():
				break
			action, node = node.select(self.c_puct)
			board.do_move(action)

		action_probs, _ = self.policy(board)
		result = board.game_end()
		if result == 0:  # Unfinished game
			node.expand(action_probs)
		node.update_recursive(result)


	def simulation(self, board, temp=1e-3):
		for n in range(self.n_playout):
			board_copy = copy.deepcopy(board)
			self.playout(board_copy)

		act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
		acts, visits = zip(*act_visits) # Unzip
		act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
		return acts, act_probs


	def update_with_move(self, last_move):
		if last_move in self.root.children:
			self.root = self.root.children[last_move]
			self.root.parent = None
		else:
			self.root = TreeNode(None, 1.0)


	def __str__(self):
		return "MCTS"				