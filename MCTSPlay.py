import Config as cfg
import MCTS

import numpy as np 
import copy

'''
Self-defined policy value calculator, which takes the current board of the Sudoku, and then analyzes the possibility of filling each individual cell
on the current board by using 1 divided by the possible numbers that could be filled in that space, and then output it as a pack of actions (the number to fill)
and action_probs (the probability of filling that number in)
'''
def policy_val_func(board):
	# a function that takes in board state and outputs a list of (action, probability) pairs and a score for the board_state
	action_probs = []
	actions = []
	if len(board.empty_positions) == 0:
		return zip(actions, action_probs), 1  #solved

	pos_prob = 1.0 / len(board.empty_positions)
	for index in list(board.empty_positions):
		row, col = board.index_to_pos(index, cfg.SUDOKU_SIZE)
		cell_idx = board.pos_to_index(row // cfg.CELL_SIZE, col // cfg.CELL_SIZE, cfg.CELL_SIZE)

		value_idx_list = [i for i in range(cfg.SUDOKU_SIZE)]
		for val_idx in range(cfg.SUDOKU_SIZE):
			if (board.analyzed_rows[row][val_idx] > 0 or board.analyzed_cols[col][val_idx] > 0
				or board.analyzed_cells[cell_idx][val_idx] > 0):
				value_idx_list.remove(val_idx)
		
		if len(value_idx_list) > 0:  #failed
			val_prob = 1.0 / len(value_idx_list)
			for val_idx in list(value_idx_list):
				actions.append(board.pack_moveID(index, val_idx))
				action_probs.append(pos_prob * val_prob)
	return zip(actions, action_probs), 0

'''
Implementation of the MCTS tree search
'''
class MCTSPlay(object):
	def __init__(self, policy_value_function, c_puct = 5, n_playout = 10000):
		self.mcts = MCTS.MCTS(policy_value_function, c_puct, n_playout)


	def reset(self):
		self.mcts.update_with_move(-1)


	def get_action(self, board, temp = 1e-3, return_prob = False):
		if return_prob:
			move_probs = np.zeros(cfg.TOTAL_ACTIONS)
		if len(board.empty_positions) > 0:
			acts, probs = self.mcts.simulation(board, temp)
			if return_prob:
				move_probs[list(acts)] = probs
			move = np.random.choice(acts, p = probs)
			self.mcts.update_with_move(move)

			if return_prob:
				return move, move_probs
			else:
				return move
		else:
			self.reset()
			print("WARNING: the board is full")


	def train_play(self, sudoku_str, board, temp = 1e-3):
		channel_states, mcts_probs = [], []
		init_state = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int)
		if (len(sudoku_str) == cfg.TOTAL_GRIDS and sudoku_str.isdigit()):
			board.parse_sudoku_str(sudoku_str, init_state)
			if board.init_board(init_state):  # valid sudoku string line
				while True:
					move, move_probs = self.get_action(board, temp = temp, return_prob = True)
					# store data
					channel_states.append(board.get_channel_state())
					mcts_probs.append(move_probs)
					# perform a move
					board.do_move(move)
					result = board.game_end()
					if result != 0:   # game end
						self.reset()
						return True, zip(channel_states, mcts_probs, [result])

		return False, zip(channel_states, mcts_probs, [False])



