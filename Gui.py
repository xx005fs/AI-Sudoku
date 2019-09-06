'''
Graphics User Interface for the Sudoku Game, that way the user can manually input a board by clicking the blocks on the screen and typing in a number
or they could also type in a sudoku string that's either formatted as 81 continuous numbers or with spaces between 3 numbers with a total of 81 numbers
'''
import Config as cfg
import MCTS
import MCTSPlay
import PolicyNet

import pickle
import numpy as np 
import tkinter as tk

SIDELENGTH = (cfg.WIN_WIDTH - cfg.MARGIN * 2) / cfg.SUDOKU_SIZE

class GUI(object):
	def __init__(self, board):
		self.board = board
		self.inputstate = True
		self.row, self.col = -1, -1
		self.init_state = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int)


	def run(self):
		if not self.inputstate:	
			result = self.run_game(None)
			if (result == 1):
				print("Congradulations! Sudoku solved!")
			else:
				print("Game Over! Failed!")
		else:
			print("Please confirm the board input first!")


	def confirm_input(self):
		sudoku_str = self.entry.get()
		sudoku_str = sudoku_str.replace(" ","")
		str_len = len(sudoku_str)
		if str_len == 0:   #Grid input
			if self.board.init_board(self.init_state):
				self.inputstate = False
		else:     #Number string input
			if str_len == cfg.TOTAL_GRIDS and sudoku_str.isdigit():
			 	self.board.parse_sudoku_str(sudoku_str, self.init_state)
			 	if self.board.init_board(self.init_state):
			 		self.inputstate = False
			 		self.draw_sudoku()
			else:
				print("Wrong Input: must be 0 ~ 9 (0 for puzzle), len = ", num_len)
				return
		if self.inputstate:	
			print("Wrong Input: not a valid Sudoku")
		else:
			return


	def graphics(self, board):
		window = tk.Tk()
		window.title("Sudoku")

		self.cv = tk.Canvas(window, height=cfg.WIN_HEIGHT, width=cfg.WIN_WIDTH, bg = 'white')
		self.cv.grid(row = 0, column = 0, columnspan = 5)
	
		self.draw_grid()
		self.cv.bind("<Button-1>", self.cell_clicked)
		self.cv.bind("<Key>", self.key_pressed)
		
		label = tk.Label(window, text="Input numbers:")
		label.grid(row = 1, column = 0, sticky = tk.SW, pady = 6)
		self.entry = tk.Entry(window, bd = 2, width = 90)
		self.entry.grid(row = 1, column = 1, columnspan = 4, sticky = tk.SW, pady = 6)

		button = tk.Button(window, text = "Clear", command = self.clear)
		button.grid(row = 2, column = 4, sticky = tk.NW, padx = 8, pady = 2)

		button1 = tk.Button(window, text = "Start!", width = 10, command = self.run)
		button1.grid(row = 3, column = 3, sticky = tk.W, padx = 2, pady = 8)

		button2 = tk.Button(window, text = "Confirm", command = self.confirm_input)
		button2.grid(row = 2, column = 3, sticky = tk.NE, padx = 8, pady = 2)

		window.mainloop()


	def draw_grid(self):
		for i in range(cfg.SUDOKU_SIZE + 1):
			if i % cfg.CELL_SIZE == 0:
				color = "black"
				thickness = 3
			else:
				color = "gray"
				thickness = 1
			x0 = cfg.MARGIN + i * SIDELENGTH
			y0 = cfg.MARGIN
			x1 = cfg.MARGIN + i * SIDELENGTH
			y1 = cfg.WIN_HEIGHT - cfg.MARGIN
			self.cv.create_line(x0, y0, x1, y1, fill = color, width = thickness)
			x0 = cfg.MARGIN
			y0 = cfg.MARGIN + i * SIDELENGTH
			x1 = cfg.WIN_WIDTH - cfg.MARGIN
			y1 = cfg.MARGIN + i * SIDELENGTH
			self.cv.create_line(x0, y0, x1, y1, fill = color, width = thickness)


	def draw_sudoku(self):
		self.cv.delete("nums")
		for i in range(cfg.SUDOKU_SIZE):
			for j in range(cfg.SUDOKU_SIZE):
				if (self.inputstate):
					val = self.init_state[i][j]
					color = "black"
				else:
					val = self.board.cur_state[i][j]
					color = "black" if self.init_state[i][j] != 0 else "blue"
				if val != 0:
					x = cfg.MARGIN + j * SIDELENGTH + SIDELENGTH / 2
					y = cfg.MARGIN + i * SIDELENGTH + SIDELENGTH / 2
					self.cv.create_text(x, y, text = val, tags = "nums", fill = color, font=("Arial", 28))


	def clear(self):
		self.inputstate = True
		self.board.clear_board()
		self.init_state = np.zeros((cfg.SUDOKU_SIZE, cfg.SUDOKU_SIZE), dtype = int)
		self.entry.delete(0, tk.END)
		self.draw_sudoku()


	def cell_clicked(self, event):
		if self.inputstate == False:
			return
		x, y = event.x, event.y
		if (cfg.MARGIN < x < cfg.WIN_WIDTH - cfg.MARGIN and cfg.MARGIN < y < cfg.WIN_HEIGHT - cfg.MARGIN):
			self.cv.focus_set()

			self.row, self.col = int((y - cfg.MARGIN) / SIDELENGTH), int((x - cfg.MARGIN) / SIDELENGTH)
		else:
			self.row, self.col = -1, -1

		self.draw_cursor()


	def draw_cursor(self):
		self.cv.delete("cursor")
		if self.row >= 0 and self.col >= 0:
			x0 = cfg.MARGIN + self.col * SIDELENGTH + 1
			y0 = cfg.MARGIN + self.row * SIDELENGTH + 1
			x1 = cfg.MARGIN + (self.col + 1) * SIDELENGTH - 1
			y1 = cfg.MARGIN + (self.row + 1) * SIDELENGTH - 1
			self.cv.create_rectangle(x0, y0, x1, y1, outline = "red", tags = "cursor")


	def key_pressed(self, event):
		if self.inputstate == False:
			return
		if self.row >= 0 and self.col >= 0 and event.char in "1234567890":
			self.init_state[self.row][self.col] = int(event.char)
			self.draw_sudoku()
			self.col, self.row = -1, -1
			self.draw_cursor()


	def run_game(self, model_file):
		if model_file:
			policy_param = pickle.load(open(cfg.MODEL_PATH, 'rb'))
			mcts_play = MCTSPlay.MCTSPlay(PolicyValueNet.PolicyValueNet(policy_param).policy_val_func, 
				c_puct = 5, n_playout = 200)
		else:
			mcts_play = MCTSPlay.MCTSPlay(MCTSPlay.policy_val_func, c_puct = 5, n_playout = 400)

		while True:
			move = mcts_play.get_action(self.board, temp = 1e-3, return_prob = False)
			# perform a move
			self.board.do_move(move)
			self.draw_sudoku()
			result = self.board.game_end()
			if result != 0:
				return result