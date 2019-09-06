'''
Test the transformation of the analyzed arrays and board for the MCTS algorithm
'''
import numpy as np
import Config as cfg
import Board as bd

def run():
	try:
		mcts_probs = np.zeros(cfg.TOTAL_ACTIONS, float)

		board  = bd.Board()
		sudoku_str = "987 254 631 641 973 285 503 861 947 834 195 726 005 387 194 719 642 853 458 726 319 396 508 472 172 439 568"
		sudoku_str = sudoku_str.replace(" ","")
		init_state = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int)
		board.parse_sudoku_str(sudoku_str, init_state)
		board.init_board(init_state)
		print("original board state -----------")
		print(board.cur_state)

		channel_state = board.get_channel_state()
		print("3D board state --------------")
		print(channel_state)

		# pseudo mcts_probs
		for row in range(cfg.SUDOKU_SIZE):
			for col in range(cfg.SUDOKU_SIZE):
				for val_idx in range(cfg.SUDOKU_SIZE):
					if channel_state[val_idx][row][col] > 0:
						index = board.pos_to_index(row, col, cfg.SUDOKU_SIZE)
						move = board.pack_moveID(index, val_idx)
						mcts_probs[move] = 0.01

		print("restore state ---------------")
		tmp_state = board.restore_state(channel_state)
		print(tmp_state)
		print("rot90 --------------------")
		rot90_state = np.rot90(channel_state, 1, axes = (1, 2))
		# print(rot90_state)
		tmp_state = board.restore_state(rot90_state)
		print(tmp_state)
		print("flip ---------------------")
		flip_state = np.flip(channel_state, 2)
		# print(flip_state)
		tmp_state = board.restore_state(flip_state)
		print(tmp_state)


		print("MCTS probs ------------------------")
		print(mcts_probs)
		print("reshape probs ------------------------")
		prob_3D = np.reshape(mcts_probs, (cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE))
		print(prob_3D)
		print("rot90 probs -------------------------")
		rot90_prob = np.rot90(prob_3D, 1, axes = (1, 2))
		print(rot90_prob)
		# tmp_prob = np.reshape(rot90_prob, -1)
		# print_prob(tmp_prob)
		print("flip probs -------------------------")
		flip_prob = np.flip(prob_3D, 2)
		print(flip_prob)
		tmp_prob = np.reshape(flip_prob, -1)
		print(tmp_prob)

		
	except KeyboardInterrupt:
		print('\nQuit')

'''
		array = []
		for i in range(1, 28):
			array.append(i)
		print("1-----", array)
		print("2-----", np.reshape(array, (3, 3, 3)))
		#print("3-----", np.rot90(np.reshape(array, (3, 3, 3)), 1, axes = (1, 2)))
		print("4-----", np.flip(np.reshape(array, (3, 3, 3)), 2))
		print("0-----", array)

		num = 1
		array1 = np.zeros((3, 3, 3), dtype = int)
		for i in range(3):
			for j in range(3):
				for k in range(3):
					array1[i][j][k] = num
					num += 1

		#print("array 1---------------")
		#print(array1)
'''


if __name__ == '__main__':
	run()