# 4 gewinnt
"""
Spieler 1 schreibt "1"
Spieler -1 schreibt "2"
"""

#board
ROWS = 4
COLUMS = 4

class Game:
	def __init__(self):
		self.player = 1
		self.board = self.init_board()
		
	def start_game(self):
		'''
		runs the game
		'''
		self.write()
		status = True
		while (status):
			if self.player == 1:
				print("Spieler 1 ist dran")
			else:
				print("Spieler 2 ist dran")
			self.make_move(self.get_input())
			self.write()
			
			x = self.check_win()
			if (x == 1):
				print("Spieler 1 hat gewonnen!!!")
				status = False
			if (x == -1):
				print("Spieler 2 hat gewonnen!!!")
				status = False
		print("Thanks for playing '12 wins'!")
	
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
		
		
		
		
	def write(self):
		'''
		gibt das Spielfeld aus
		'''
		for i in range(ROWS):
			string = "|"
			for j in range(COLUMS):
				string += " " + str(self.board[j][i]) + " |"
				### besseres graphic design mit sternchen anstatt nullen
				"""
				string += " " + str(self.graphic(j,i)) + " |"
				"""
				###
			print(string)
		print()

		
		"""
	def graphic(self,j,i):
		x = self.board[j][i]
		if x == 0:
			x = "*"
		return x
		"""
		
	
	
	def make_move(self,col):
		'''
		setzt einen Stein
		'''
		row = self.test_validity(col)
		
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
		
	def test_validity(self,col):
		'''
		returnt die oberste freie Reihe
		wenn alle Reihen besetzt sind, dann wird -1 returnt
		'''
		row = ROWS-1

		while ((self.board[col][row] != 0) and (row >= 0)):
			row -= 1
		return row
		
		
		
	def check_win(self):
		'''
		Ueberprueft, ob ein Spieler bereits gewonnen hat
		'''
		x = self.check_row() + self.check_colum() + self.check_diagonal_bottom_left() + self.check_diagonal_bottom_right()
		if (x > 0):
			x = 1
		if (x < 0):
			x = -1
		return x
		
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
			
	def check_diagonal_bottom_left(self):
		'''
		ueberprueft, ob eine der Diagonalen von links unten nach rechts oben 4 gleiche Symbole hat
		'''
		if (COLUMS >= 4) and (ROWS >= 4):
			# die Diagonalen von links unten bis rechts oben ueberpruefen
			for i in range(COLUMS-3):
				list = []
				colum = i
				row = ROWS-1
				while (colum < COLUMS and row >= 0):
					list.append(self.board[colum][row])
					colum+=1
					row-=1
				x = self.check_list(list)
				if x != 0:
					return x
			for i in range(ROWS-4):
				list = []
				colum = 0
				row = ROWS-1-i
				while (colum < COLUMS and row >= 0):
					list.append(self.board[colum][row])
					colum+=1
					row-=1
				x = self.check_list(list)
				if x!= 0:
					return x
			return 0
	
	def check_diagonal_bottom_right(self):
		'''
		ueberprueft, ob eine der Diagonalen von rechts unten nach links oben 4 gleiche Symbole hat
		'''		
		if (COLUMS >= 4) and (ROWS >= 4):
			for i in range(COLUMS-3):
				list = []
				colum = COLUMS-1-i
				row = ROWS-1
				while (colum >= 0 and row >= 0):
					list.append(self.board[colum][row])
					colum-=1
					row-=1
				x = self.check_list(list)
				if x != 0:
					return x
			for i in range (ROWS-4):
				list = []
				colum = COLUMS-1
				row = ROWS-2-i
				while (colum >= 0 and row >= 0):
					list.append(self.board[colum][row])
					colum-=1
					row-=1
				x = self.check_list(list)
				if x!= 0:
					return x
			return 0
		


		
		
	
game = Game()
game.start_game()