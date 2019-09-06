import Config as cfg
import numpy as np

class Board(object):
	def __init__(self):
		self.empty_positions = []  # empty position by indices
		self.cur_state = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int) # current board state
		self.analyzed_rows = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int) # an array that stores which numbers can be put into all rows
		self.analyzed_cols = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int) # an array that stores which numbers can be put into all columns
		self.analyzed_cells = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int) # an array that stores which numbers can be put into all cells

'''
Clears all the analyzed states and the current state board for the next step
'''
	def clear_board(self):
		self.cur_state = [[0 for j in range(cfg.SUDOKU_SIZE)] for i in range(cfg.SUDOKU_SIZE)]
		self.empty_positions.clear()
		self.analyzed_rows = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int)
		self.analyzed_cols = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int)
		self.analyzed_cells = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int)


	def init_board(self, board_state):
		self.last_move = -1
		self.clear_board()
		for row in range(cfg.SUDOKU_SIZE):
			for col in range(cfg.SUDOKU_SIZE):
				self.cur_state[row][col] = board_state[row][col]
		return self.state_analyze()

'''
Parses the Sudoku string that's typed in by the user into the board, and then displaying it on the GUI screen
'''
	def parse_sudoku_str(self, sudoku_str, board_state):
		row, col = 0, 0
		for x in list(sudoku_str):
			board_state[row][col] = int(x)
			col += 1
			if (col == cfg.SUDOKU_SIZE):
				col = 0
				row += 1

'''
Converting the index of a number on the board into row and column to represent location of a certain block
'''
	def index_to_pos(self, index, dimension):
		row = index // dimension
		col = index % dimension
		return row, col

'''
Converting the position back to index with row and cols
'''
	def pos_to_index(self, row, col, dimension):
		return int(row * dimension + col)

'''
Convert the block value into move_ID, that is the packed value that's calculated to be the location of the block in index form added onto the value multiplied by the total grids value
The val_idx are values that could be filled into this block
'''
	def pack_moveID(self, pos_idx, val_idx):
		return int(val_idx * cfg.TOTAL_GRIDS + pos_idx)

'''
Unpacks the moveID
'''
	def unpack_moveID(self, moveID):
		val_idx = moveID // cfg.TOTAL_GRIDS
		pos_idx = moveID % cfg.TOTAL_GRIDS
		return pos_idx, val_idx

'''
Analyzes and writes all the analyzed information into the analyzed_rows, analyzed_cols, and analyzed_cells arrays, that way the functions
can use them for MCTS and find the solution to the boards
'''
	def state_analyze(self):
		for row in range(cfg.SUDOKU_SIZE):
			for col in range(cfg.SUDOKU_SIZE):
				val = self.cur_state[row][col]
				if val != 0:
					if not (val > 0 and val <= cfg.SUDOKU_SIZE):
						return False

					val_idx = val - 1
					# Analyze Rows
					if self.analyzed_rows[row][val_idx] > 0: # Detects Duplicates
						return False
					else:
						self.analyzed_rows[row][val_idx] = 1
					# Analyze Columns
					if self.analyzed_cols[col][val_idx] > 0: # Detects Duplicates
						return False
					else:
						self.analyzed_cols[col][val_idx] = 1
					# Analyze 3x3 cells
					cell_idx = self.pos_to_index(row // cfg.CELL_SIZE, col // cfg.CELL_SIZE, cfg.CELL_SIZE)
					if self.analyzed_cells[cell_idx][val_idx] > 0: # Detects Duplicates
						return False
					else:
						self.analyzed_cells[cell_idx][val_idx] = 1
				else:
					self.empty_positions.append(self.pos_to_index(row, col, cfg.SUDOKU_SIZE))
					
		return True

'''
Parses moveID and does the operation needed (say for example filling a value of 6 into block at row 2 col 5)
'''
	def do_move(self, move):
		pos_idx, val_idx = self.unpack_moveID(move)
		row, col = self.index_to_pos(pos_idx, cfg.SUDOKU_SIZE)
		cell_idx = self.pos_to_index(row // cfg.CELL_SIZE, col // cfg.CELL_SIZE, cfg.CELL_SIZE)
		self.analyzed_rows[row][val_idx] = 1
		self.analyzed_cols[col][val_idx] = 1
		self.analyzed_cells[cell_idx][val_idx] = 1
		self.cur_state[row][col] = val_idx + 1  #set value: val = val_idx + 1
		self.empty_positions.remove(pos_idx)
		self.last_move = move

'''
Quits the game when this is satisfied, and also tells the neural network whether the sudoku is solved, unfinished, or just simply failed
The three states are 1, 0, and -1, representing solved, unfinished, and failed
'''
	def game_end(self):
		if len(self.empty_positions) == 0:
			return 1  # solved

		for index in list(self.empty_positions):
			row, col = self.index_to_pos(index, cfg.SUDOKU_SIZE)
			cell_idx = self.pos_to_index(row // cfg.CELL_SIZE, col // cfg.CELL_SIZE, cfg.CELL_SIZE)

			value_idx_list = [i for i in range(cfg.SUDOKU_SIZE)]
			for val_idx in range(cfg.SUDOKU_SIZE):
				if self.analyzed_rows[row][val_idx] > 0 or self.analyzed_cols[col][val_idx] > 0 or self.analyzed_cells[cell_idx][val_idx] > 0:
					value_idx_list.remove(val_idx)

			if len(value_idx_list) == 0:
				return -1 # failed

		return 0  # unfinished

'''
Used for the MCTS tree search, that way it can list out all the possible actions that it can do for the next step and pick one out of random and actually execute it
'''
	def get_available_actions(self):
		available_actions = []
		for index in list(self.empty_positions):
			row, col = self.index_to_pos(index, cfg.SUDOKU_SIZE)
			cell_idx = self.pos_to_index(row // cfg.CELL_SIZE, col // cfg.CELL_SIZE, cfg.CELL_SIZE)
			
			value_idx_list = [i for i in range(cfg.SUDOKU_SIZE)]
			for val_idx in range(cfg.SUDOKU_SIZE):
				if self.analyzed_rows[row][val_idx] > 0 or self.analyzed_cols[col][val_idx] > 0 or self.analyzed_cells[cell_idx][val_idx] > 0:
					value_idx_list.remove(val_idx)
			
			for val_idx in list(value_idx_list):
				available_actions.append(self.pack_moveID(index, val_idx))
		return available_actions	

'''
Convert the boolean datas stored in analyzed arrays into a 3D array, that way the machine can read the data instead of relying on boolean values
'''
	def get_channel_state(self):
		channel_state = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = float)
		# total available actions
		for row in range(cfg.SUDOKU_SIZE):
			for col in range(cfg.SUDOKU_SIZE):
				val = self.cur_state[row][col]
				if val in range(1, cfg.SUDOKU_SIZE + 1):
					channel_state[val - 1][row][col] = 1.0
		return channel_state

'''
Converts back from channel state to original way to represent data
'''
	def restore_state(self, channel_state):  # for test only
		state = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int)
		for val_idx in range(cfg.SUDOKU_SIZE):
			for row in range(cfg.SUDOKU_SIZE):
				for col in range(cfg.SUDOKU_SIZE):
					if channel_state[val_idx][row][col] > 0:
						state[row][col] = val_idx + 1
		return state

'''
Debugging code
'''
	def debug_analyze(self):
		print("---------------row---------------")
		print(self.analyzed_rows)
		print("\n\n\n---------------col---------------")
		print(self.analyzed_cols)
		print("\n\n\n---------------cell---------------")
		print(self.analyzed_cells)
		print("\n\n\n---------------empties---------------")
		print(self.empty_positions)