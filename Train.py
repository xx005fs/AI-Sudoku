'''
Train the model. Implemented and modified from https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/train.py
'''
from __future__ import print_function
import Config as cfg
import Board
import MCTS
import MCTSPlay
import PolicyNet

import random
import numpy as np
from collections import defaultdict, deque

class TrainPipeline(object):
	def __init__(self, model_file = None):
		self.board = Board.Board()

		# Training parameters
		self.learn_rate = 2e-3
		self.temp = 1.0
		self.n_playout = 400
		self.c_puct = 5
		self.buffer_size = 10000
		self.batch_size = 512   # min batch size for training
		self.lr_multiplier = 1.0
		self.data_buffer = deque(maxlen = self.buffer_size)
		self.play_batch_size = 1
		self.epochs = 5 # num of train steps for each update
		self.kl_targ = 0.02
		self.check_freq = 50
		self.game_batch_num = 1500
		self.best_solved_ratio = 0.0

		if model_file:  # start training from an initial policy-value net
			self.policy_value_net = PolicyNet.PolicyValueNet(model_file = model_file)
			self.MCTS_player = MCTSPlay.MCTSPlay(self.policy_value_net.policy_val_func, 
				c_puct = self.c_puct, n_playout = self.n_playout)
		else:  # start training from a new policy-value net
			self.policy_value_net = PolicyNet.PolicyValueNet(model_file = None)		
			self.MCTS_player = MCTSPlay.MCTSPlay(MCTSPlay.policy_val_func, c_puct = self.c_puct, 
				n_playout = self.n_playout)
		

	def get_equi_data(self, play_data): # handling rotation and flipping
		# play_data: [(state, MCTS_prob, result), (...), (...)]
		extend_data = []
		for state, mcts_prob, result in play_data:
			for i in [1, 2, 3, 4]:
				# rotate counter-clockwise
				equi_state = np.rot90(state, i, axes = (1, 2))
				equi_mcts_prob = np.rot90(np.reshape(mcts_prob, 
					(cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE)), i, axes = (1, 2))
				equi_mcts_prob = np.reshape(equi_mcts_prob, -1)
				extend_data.append((equi_state, equi_mcts_prob, result))

				# flip horizontally
				equi_state = np.flip(state, axis = 2)
				equi_mcts_prob = np.flip(np.reshape(equi_mcts_prob, 
					(cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE)), axis = 2)
				equi_mcts_prob = np.reshape(equi_mcts_prob, -1)
				extend_data.append((equi_state, equi_mcts_prob, result))
		return extend_data


	def collect_play_data(self, sudoku_str, n_games = 1): # play games and collect data for training
		for i in range(n_games):
			result, game_data = self.MCTS_player.train_play(sudoku_str, self.board, temp = self.temp)
			if result:
				play_data = list(game_data)[:]
				self.episode_len = len(play_data)

				# extend data with rotation and flipping
				play_data = self.get_equi_data(play_data)
				self.data_buffer.extend(play_data)


	def policy_update(self):
		min_batch = random.sample(self.data_buffer, self.batch_size)
		state_batch = [data[0] for data in min_batch]
		mcts_probs_batch = [data[1] for data in min_batch]
		result_batch = [data[2] for data in min_batch]

		old_probs, old_result = self.policy_value_net.policy_value(state_batch)
		for i in range(self.epochs):
			loss, entropy = self.policy_value_net.train_step(state_batch, 
				mcts_probs_batch, result_batch, self.learn_rate * self.lr_multiplier)
			new_probs, new_reuslt = self.policy_value_net.policy_value(state_batch)
			kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis = 1))
			if kl > self.kl_targ * 4:  # stop early if D_KL diverges badly
				break

		# adjust the learning rate
		if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
			self.lr_multiplier /= 1.5
		elif kl < self.kl_targ / 2 and self.lr_multiplier < 10.0:
			self.lr_multiplier *= 1.5

		explained_var_old = (1 - np.var(np.array(result_batch) - old_result.flatten()) 
			/ np.var(np.array(result_batch)))
		explained_var_new = (1 - np.var(np.array(result_batch) - new_reuslt.flatten()) 
			/ np.var(np.array(result_batch)))

		print(("kl:{0:5.3f},"
			"lr_multiplier:{1:5.3f},"
			"loss:{2:5.3f},"
			"entropy:{3:5.3f},"
			"explained_var_old:{4:5.3f},"
			"explained_var_new:{5:5.3f}").format(kl, self.lr_multiplier, loss, entropy, 
			explained_var_old, explained_var_new))
		return loss, entropy


	def run(self):
		try:
			train_path = cfg.TRAIN_PATH
			file = open(train_path, 'r')
			try:
				lines = file.read().splitlines()[1:]
				file.close()
				game_batch_num = 1
				for line_str in list(lines):
					sudoku_str, _ = line_str.split(",")
					self.collect_play_data(sudoku_str, self.play_batch_size)
					print("steps :{}, episode_len:{}".format(game_batch_num, self.episode_len))

					if len(self.data_buffer) > self.batch_size:
						loss, entropy = self.policy_update()

					# check the performance of the current model and save the model parameters
					if ((game_batch_num) % self.check_freq) == 0:
						print("current mcts_play batch:{}".format(game_batch_num))
						self.policy_value_net.save_model(cfg.MODEL_PATH)

					game_batch_num += 1
					if game_batch_num > self.game_batch_num:
						 break							
			except KeyboardInterrupt:
				print("\n\rquit")
			
		except IOError:
			print("File doesn't exist")
		

if __name__ == "__main__":
	pipeline = TrainPipeline()
	# pipeline = TrainPipeline(cfg.MODEL_PATH)
	pipeline.run()

