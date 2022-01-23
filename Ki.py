# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:41:59 2022

@author: PhiDM
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os
import random
import copy
from torch.utils.tensorboard import SummaryWriter



#exchange
PLAYER1 = 1
PLAYER2 = -1
COLUMS = 7
ROWS = 6

class Game():
    def __init__(self):
        self.field = []
        for col in range(COLUMS):
            single_row = []
            for row in range(ROWS):
                single_row.append(0)
            self.field.append(single_row)
    def make_move(self,game,move,player):
        for idx in range(len(game.field[move])):
            if game.field[move][idx] == 0:
                game.field[move][idx] = player
                return game
        return game
    
    def switch_player(self,player):
        if player == PLAYER1:
            player = PLAYER2
        elif player == PLAYER2:
            player = PLAYER1
        return player
    
    def draw(self,game):
        for col in range(COLUMS):
            for row in range(ROWS):
                if game.field[col][row] == 0:
                    return False
        return True
    
    def won(self,game):
        for row in range(ROWS):
            for col in range(COLUMS-3):
                if game.field[col][row] == game.field[col+1][row] and game.field[col+1][row] == game.field[col+2][row] and game.field[col+2][row] == game.field[col+3][row]:
                    return game.field[col][row]
        for col in range(COLUMS):
            for row in range(ROWS-3):
                if game.field[col][row] == game.field[col][row+1] and game.field[col][row+1] == game.field[col][row+2] and game.field[col][row+2] == game.field[col][row+3]:
                    return game.field[col][row]
        for col in range(COLUMS-3):
            for row in range(ROWS-3):
                if game.field[col][row] == game.field[col+1][row+1] and game.field[col+1][row+1] == game.field[col+2][row+2] and game.field[col+2][row+2] == game.field[col+3][row+3]:
                    return game.field[col][row]
        for col in range(COLUMS-3):
            for row in range(ROWS-3):
                if game.field[col+3][row] == game.field[col+2][row+1] and game.field[col+2][row+1] == game.field[col+1][row+2] and game.field[col+1][row+2] == game.field[col][row+3]:
                    return game.field[col][row]

                




epoch = 0


class Batch():
    def __init__(self):
        self.mem = []
        self.batch_idx = 0
        self.BATCHSIZE = 50

    def addToMem(self,state,next_state,move,player,reward):
        self.mem.append([state,next_state,move,player,reward])
        self.batch_idx = self.batch_idx + 1
	
    def updateNextState(self,state):
        self.mem[self.batch_idx-1][1] = state
	
    def editEndingReward(self,reward,change_for_second_player):
        self.mem[self.batch_idx-1][4] = copy.deepcopy(reward)
        self.mem[self.batch_idx-2][4] = copy.deepcopy(reward) * copy.deepcopy(change_for_second_player)
    
    def reset_batch(self):
        self.mem = []
        self.batch_idx = 0
    
    
class Trainer():
    def __init__(self):
        self.WINNINGREWARD = 10
        self.DRAWREWARD = 0.5
        self.INVALID_MOVE_REWARD = -100
        self.GAMMA = 0.8
        self.memory = Batch()
        self.model = Ki().initialize_model()
        self.writer = SummaryWriter('models/tensorboard')
        self.plotting_step = 0
        
    def random_game(self):
        game = Game()
        player = PLAYER1
        free_slots = []
        for col in range(COLUMS):
            free_slots.append(ROWS)
        active = True
        while active == True:
            old_game = copy.deepcopy(game.field)
            move = self.model.random_move(free_slots)
            game = game.make_move(game,move,player)
            free_slots[move] = free_slots[move] - 1
    		# won() gibt das Zeichen des Spielers zurück, der gewonnen hat
            reward = 0
            self.memory.addToMem(old_game,None,move,player,reward)
            if game.won(game) != 0:
                reward = self.WINNINGREWARD
                self.memory.editEndingReward(reward,-1)
                active = False
                print("rW")
            elif game.draw(game) == True:
                reward = self.DRAWREWARD
                self.memory.editEndingReward(reward,1)
                active = False
                print("rD")
            else:
                self.memory.updateNextState(copy.deepcopy(game.field))
            player = game.switch_player(player)

    def hybrid(self):
        offset = random.randint(0,1)
        game = Game()
        player = PLAYER1
        free_slots = []
        for col in range(COLUMS):
            free_slots.append(ROWS)
        active = True
        while active == True:
            old_game = copy.deepcopy(game.field)
            input_game = copy.deepcopy(game.field)
            if offset == 1:
                move = self.model.model_move(self.model.convert_game(input_game,player))
            if offset == 0:
                move = self.model.random_move(free_slots)
            game = game.make_move(game,move,player)
            free_slots[move] = free_slots[move] - 1
    		# won() gibt das Zeichen des Spielers zurück, der gewonnen hat
            reward = 0
            self.memory.addToMem(old_game,None,move,player,reward)
            if game.won(game) != 0:
                reward = self.WINNINGREWARD
                self.memory.editEndingReward(reward,-1)
                active = False
                print("hW")
            elif game.draw(game) == True:
                reward = self.DRAWREWARD
                self.memory.editEndingReward(reward,1)
                active = False
                print("hD")
            elif game.field == input_game:
                reward = self.INVALID_MOVE_REWARD
                self.memory.editEndingReward(reward, 0)
                active = False
                print("hI")
            else:
                self.memory.updateNextState(copy.deepcopy(game.field))
            #print(game)
            player = game.switch_player(player)
            offset = offset + 1
            offset = offset % 2

    def ki_only_game(self):
        game = Game()
        player = PLAYER1
        active = True
        while active == True:
            old_game = copy.deepcopy(game.field)
            input_game = copy.deepcopy(game.field)
            move = self.model.model_move(self.model.convert_game(input_game,player))
            game = game.make_move(game,move,player)
    		# won() gibt das Zeichen des Spielers zurück, der gewonnen hat
            reward = 0
            self.memory.addToMem(old_game,None,move,player,reward)
            if game.won(game) != 0:
                reward = self.WINNINGREWARD
                self.memory.editEndingReward(reward,-1)
                active = False
                print("kW")
            elif game.draw(game) == True:
                reward = self.DRAWREWARD
                self.memory.editEndingReward(reward,1)
                active = False
                print("kD")
            elif game.field == input_game:
                reward = self.INVALID_MOVE_REWARD
                self.memory.editEndingReward(reward, 0)
                active = False
                print("kI")
            else:
                self.memory.updateNextState(copy.deepcopy(game.field))
            #print(game)
            player = game.switch_player(player)
        
    def train_step(self,sample,iteration,mode):
        state = sample[0]
        next_state = sample[1]
        move = sample[2]
        player = sample[3]
        reward = sample[4]
        state = self.model.convert_game(state,player)
        if next_state is None and reward == 0:
            target = model(state)
        elif next_state is None:
            target = self.model(state)
            target[move] = target[move] + self.GAMMA * reward
        else:
            next_state = self.model.convert_game(next_state,player)
            target = self.model(state)
            target[move] =  -(self.GAMMA * max(self.model(next_state).tolist()))
        prediction = self.model(state)
	
        loss = self.model.loss_func(prediction,target)
        print(loss)
    
        self.writer.add_scalar(mode, loss, iteration)
    
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
	
	
    def train(self,iterations,mode):
        global epoch
        for iteration in range(iterations):
            for samples in range(self.memory.BATCHSIZE):
                if mode == 'r':
                    self.random_game()
                elif mode == 'k':
                    self.ki_only_game()
                elif mode == 'h':
                    self.hybrid()
            for samples in range(self.memory.batch_idx):
                self.train_step(self.memory.mem[self.memory.batch_idx-samples-1],self.plotting_step,mode)
                self.plotting_step = self.plotting_step + 1
            self.memory.reset_batch()
            epoch = epoch + 1
            self.model.save_model()
            print(epoch)
			
	


class Ki(nn.Module):
    def __init__(self):
        super(Ki, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(42, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512, 7),
        )
        
        self.epoch = 0
        self.LR = 0.8
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr = self.LR)
        self.PATH = 'models/model.pth'
    
    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
            }, self.PATH)
        
    def initialize_model(self):
        global model
        if os.path.isfile(self.PATH):
            checkpoint = torch.load(self.PATH)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.eval()
        else:
            self = Ki()
        return self
    
    def convert_game(self,game,player):
    	#return game with empty spaces as 0, the Ki's stones as 1 and the enemys stones as -1
        if player == PLAYER2:
            for col in range(COLUMS):
                for row in range(ROWS):
                    if game[col][row] == 1:
                        game[col][row] = -1
                    elif game[col][row] == -1:
                        game[col][row] = 1
        game = torch.Tensor(game)
        game = torch.flatten(game)
        return game
	
	
    def random_move(self,free_slots):
        free = copy.deepcopy(free_slots)
        deleted_colums = 0
        for col in range(COLUMS):
            actual_col = col - deleted_colums
            if free[actual_col] != 0:
                free[actual_col] = col
            else:
                free.pop(actual_col)
                deleted_colums = deleted_colums + 1
        picked_col = random.randint(0, len(free)-1)
        move = free[picked_col]
        return move

    def model_move(self,game):
        move = self(torch.Tensor(game))
        move = torch.argmax(move).item()
        return move






#EPS_START = 0.1
#EPS_END = 0.98
#EPS_STEPS = 25000





        
training = Trainer()
for i in range(10):
    training.train(20,'r')
    training.train(30,'h')
    training.train(40,'k')
    training.model.save_model()
    training.model.LR = training.model.LR / 2

training.writer.add_graph(training.model, input_to_model=training.model.convert_game(Game().field,PLAYER1), verbose=True, use_strict_trace=True)
training.writer.close()