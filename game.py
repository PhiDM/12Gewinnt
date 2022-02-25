# 4 gewinnt
"""
Spieler 1 schreibt "1"
Spieler -1 schreibt "2"
"""

#board
ROWS = 6
COLUMS = 7
STATUS = True

class Game:

	def test(self):
		print(self.check_colum())
		print(self.check_colum())

		
	def start_game(self):
		'''
		runs the game
		'''
		self.write()
		for i in range(10):
			if self.player == 1:
				print("Spieler 1 ist dran")
			else:
				print("Spieler 2 ist dran")
			self.make_move(self.get_input())
			self.write()
			
			print(self.check_row())
		
		
		
	def check_win(self):
		'''
		Ueberprueft, ob ein Spieler bereits gewonnen hat
		'''
		return STATUS
		
		
	def check_list(self,list):
		'''
		ueberprueft,ob in einer Liste vier gleiche, aufeinanderfolgende Symbole enthalten sind
		diese Symbole sind ungleich 0
		'''
		counter = 0
		for i in range(len(list)):
			if list[i] == 0:
				counter = 0
			if list[i] == 1:
				if counter >= 0:
					counter += 1
				else:
					counter = 1
			if list[i] == 2:
				if counter <= 0:
					counter -= 1
				else:
					counter = -1
			if counter == 4:
				return 1
			if counter == -4:
				return -1
		return 0
	
	
	def check_row(self):
		'''
		ueberprueft, ob eine der Reihen 4 gleiche Symbole hat
		'''
		for i in range(ROWS):
			list = []
			for j in range(COLUMS):
				list.append(self.board[j][i])
			x = self.check_list(list)
			if x != 0:
				return x
		return 0
				
	
	def check_colum(self):
		'''
		ueberprueft,ob eine der Spalten 4 gleiche Symbole hat
		'''
		for i in range(COLUMS):
			x = self.check_list(self.board[i])
			if x != 0:
				return x
		return 0
			
		
	
	def check_diagonal(self):
		'''
		ueberprueft, ob eine der Diagonalen 4 gleiche Symbole hat
		'''
		if (COLUMS >= 4) and (ROWS <= 4):
			# die Diagonalen von links unten bis rechts oben ueberpruefen
			for i in range((COLUMS+ROWS)-7):
				
				
			#die Diagonalen von links oben bis rechts unten ueberpruefen
			
		
		
		
	
	def write(self):
		'''
		gibt das Spielfeld aus
		'''
		for i in range(ROWS):
			string = "|"
			for j in range(COLUMS):
				string += " " + str(self.board[j][i]) + " |"
			print(string)
		print()


	
	def __init__(self):
		self.player = 1
		self.board = self.init_board()
		
#	def __str__(self):
		

	def init_board(self):
		'''
		erstellt das Spielfeld mit den angegebenen Dimensionen
		'''
		board = []
		for i in range(COLUMS):
			board.append([])
			for j in range(ROWS):
				board[i].append(0)
		return board
	
	def test_validity(self,col):
		'''
		returnt die oberste freie Reihe
		wenn alle Reihen besetzt sind, dann wird -1 returnt
		'''
		row = ROWS-1

		while ((self.board[col][row] != 0) and (row >= 0)):
			row -= 1
		return row
		
		
	def make_move(self,col):
		'''
		setzt einen Stein
		'''
		row = self.test_validity(col)
		if (row < 0):
			print("Dieser Zug ist nicht moeglich")
		else:
			if (self.player == 1):
				self.board[col][row] = 1
			else:
				self.board[col][row] = 2
			self.player = self.player * (-1)

	
	def get_input(self):
		'''
		erfragt, in welche Spalte (column) der Spieler seinen Stein setzen moechte
		returnt diesen input
		'''
		playerInput = input("In welche Spalte moechtest du deinen Stein werfen?")-1
		return playerInput
	
game = Game()
game.start_game()