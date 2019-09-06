import Gui
import Board
import MCTS

def run():
	try:
		board = Board.Board()
		gui = Gui.GUI(board)

		gui.graphics(board)
	except KeyboardInterrupt:
		print('\nQuit')

if __name__ == '__main__':
	run()