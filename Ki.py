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
import math
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import game


#exchange
PLAYER1 = game.PLAYER1
PLAYER2 = game.PLAYER2
COLUMS = game.COLUMNS
ROWS = game.ROWS

class Game():
    def __init__(self):
        self.field = game.init_board
       
    def make_move(self,game,move,player):
        for idx in range(len(game.field[move])):
            if game.field[move][idx] == 0:
                game.field[move][idx] = player
                return game
        return game
    
    #def switch_player(self,player):
     #   if player == PLAYER1:
      #      player = PLAYER2
       # elif player == PLAYER2:
        #    player = PLAYER1
        #return player
    
    #def draw(self,game):
     #   for col in range(COLUMS):
      #      for row in range(ROWS):
       #         if game.field[col][row] == 0:
        #            return False
        #return True
    
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
        self.mem = {
            "state":[],
            "next_state":[],
            "move":[],
            "player":[],
            "reward":[]
            }
        self.batch_idx = 0
        self.BATCHSIZE = 100

    def addToMem(self,state,next_state,move,player,reward):
        self.mem["state"].append(state)
        self.mem["next_state"].append(next_state)
        self.mem["move"].append(move)
        self.mem["player"].append(player)
        self.mem["reward"].append(reward)
        self.batch_idx = self.batch_idx + 1
	
    def updateNextState(self,nextstate):
        self.mem["next_state"][self.batch_idx-1] = nextstate
	
    def editEndingReward(self,reward,change_for_second_player):
        self.mem["reward"][self.batch_idx-1] = copy.deepcopy(reward)
        self.mem["reward"][self.batch_idx-2] = copy.deepcopy(reward) * copy.deepcopy(change_for_second_player)
    
    def reset_batch(self):
        self.mem = {
            "state":[],
            "next_state":[],
            "move":[],
            "player":[],
            "reward":[]
            }
        self.batch_idx = 0
    

class Replay():
    def __init__(self):
        self.mem = {
            "state":[],
            "next_state":[],
            "move":[],
            "player":[],
            "reward":[],
            "loss":[]
            }
        self.batch_idx = 0
        self.BATCHSIZE = 500

    def addToMem(self,state,next_state,move,player,reward,loss):
        if(self.batch_idx < self.BATCHSIZE):
            self.mem["state"].append(state)
            self.mem["next_state"].append(next_state)
            self.mem["move"].append(move)
            self.mem["player"].append(player)
            self.mem["reward"].append(reward)
            self.mem["loss"].append(loss)
            self.batch_idx = self.batch_idx + 1
        else:
            index = np.argmin(self.mem["loss"])
            if(loss > self.mem["loss"][index]):
                self.mem["state"][index] = state
                self.mem["next_state"][index] = next_state
                self.mem["move"][index] = move
                self.mem["player"][index] = player
                self.mem["reward"][index] = reward
                self.mem["loss"][index] = loss
    	
    

class Trainer():
    def __init__(self):
        self.WINNINGREWARD = 10
        self.DRAWREWARD = 0.5
        self.INVALID_MOVE_REWARD = -100
        self.GAMMA = 0.8
        self.EPS_START = 0.98
        self.EPS_END = 0.02
        self.EPS_STEPS = 12500
        self.moves = 0
        self.memory = Batch()
        self.replay = Replay()
        self.model = Ki().initialize_model()
        self.writer = SummaryWriter('models/tensorboard')
        self.plotting_step = 0
        
    def random_game(self):
        spiel = game.Game()
        player = PLAYER1
        free_slots = []
        for col in range(COLUMS):
            free_slots.append(ROWS)
        active = True
        while active == True:
            old_game = copy.deepcopy(spiel.board)
            move = self.model.random_move(free_slots)
            spiel = spiel.make_move(spiel,move,player)
            free_slots[move] = free_slots[move] - 1
    		# won() gibt das Zeichen des Spielers zur??ck, der gewonnen hat
            reward = 0
            self.memory.addToMem(old_game,None,move,player,reward)
            if game.won(game) != 0:
                reward = self.WINNINGREWARD
                self.memory.editEndingReward(reward,-1)
                active = False
                print("rW")
            elif spiel.draw() == True:
                reward = self.DRAWREWARD
                self.memory.editEndingReward(reward,1)
                active = False
                print("rD")
            else:
                self.memory.updateNextState(copy.deepcopy(spiel.board))
                spiel.switch_player()

    def hybrid(self):
        offset = random.randint(0,1)
        spiel = game.Game()
        player = PLAYER1
        free_slots = []
        for col in range(COLUMS):
            free_slots.append(ROWS)
        active = True
        while active == True:
            old_game = copy.deepcopy(spiel.board)
            input_game = copy.deepcopy(spiel.board)
            if offset == 1:
                flattened_game = self.model.flatten_game(self.model.convert_game(input_game,player))
                free_colums = flattened_game.pop(0)
                taken_colums = self.model.taken(free_colums)
                move = self.model.model_move(Variable(torch.Tensor(copy.deepcopy(flattened_game))),taken_colums)
            if offset == 0:
                move = self.model.random_move(free_slots)
            game = game.make_move(game,move,player)
            free_slots[move] = free_slots[move] - 1
    		# won() gibt das Zeichen des Spielers zur??ck, der gewonnen hat
            reward = 0
            self.memory.addToMem(old_game,None,move,player,reward)
            if game.won(game) != 0:
                reward = self.WINNINGREWARD
                self.memory.editEndingReward(reward,-1)
                active = False
                print("hW")
            elif spiel.draw() == True:
                reward = self.DRAWREWARD
                self.memory.editEndingReward(reward,1)
                active = False
                print("hD")
            elif spiel.board == input_game:
                reward = self.INVALID_MOVE_REWARD
                self.memory.editEndingReward(reward, 0)
                active = False
                print("hI")
            else:
                self.memory.updateNextState(copy.deepcopy(spiel.board))
            #print(game)
            spiel.switch_player()
            offset = offset + 1
            offset = offset % 2

    def ki_only_game(self):
        spiel = game.Game()
        player = PLAYER1
        active = True
        while active == True:
            old_game = copy.deepcopy(spiel.board)
            input_game = copy.deepcopy(spiel.board)
            flattened_game = self.model.flatten_game(self.model.convert_game(input_game,player))
            free_colums = flattened_game.pop(0)
            taken_colums = self.model.taken(free_colums)
            move = self.model.model_move(Variable(torch.Tensor(copy.deepcopy(flattened_game))),taken_colums)
            game = game.make_move(game,move,player)
    		# won() gibt das Zeichen des Spielers zur??ck, der gewonnen hat
            reward = 0
            self.memory.addToMem(old_game,None,move,player,reward)
            if game.won(game) != 0:
                reward = self.WINNINGREWARD
                self.memory.editEndingReward(reward,-1)
                active = False
                print("kW")
            elif spiel.draw() == True:
                reward = self.DRAWREWARD
                self.memory.editEndingReward(reward,1)
                active = False
                print("kD")
            elif spiel.board == input_game:
                reward = self.INVALID_MOVE_REWARD
                self.memory.editEndingReward(reward, 0)
                active = False
                print("kI")
            else:
                self.memory.updateNextState(copy.deepcopy(spiel.board))
            #print(game)
            spiel.switch_player()
            
    def epsilon_training(self):
        spiel = game.Game()
        player = PLAYER1
        active = True
        while active == True:
            old_game = copy.deepcopy(spiel.board)
            input_game = copy.deepcopy(spiel.board)
            threshold = self.EPS_END + (self.EPS_START-self.EPS_END) * math.exp(-1. * self.moves / self.EPS_STEPS)
            self.moves = self.moves + 1
            if random.random() > threshold:
                flattened_game = self.model.flatten_game(self.model.convert_game(input_game,player))
                free_colums = flattened_game.pop(0)
                taken_colums = self.model.taken(free_colums)
                move = self.model.model_move(Variable(torch.Tensor(copy.deepcopy(flattened_game))),taken_colums)
            else:
                move = self.model.random_move(self.model.find_free_cols(self.model.convert_game(input_game,player)))
            game = game.make_move(game,move,player)
    		# won() gibt das Zeichen des Spielers zur??ck, der gewonnen hat
            reward = 0
            self.memory.addToMem(old_game,None,move,player,reward)
            if game.won(game) != 0:
                reward = self.WINNINGREWARD
                self.memory.editEndingReward(reward,-1)
                active = False
                print("eW")
            elif spiel.draw() == True:
                reward = self.DRAWREWARD
                self.memory.editEndingReward(reward,1)
                active = False
                print("eD")
            elif spiel.board == input_game:
                reward = self.INVALID_MOVE_REWARD
                self.memory.editEndingReward(reward, 0)
                active = False
                print("eI")
            else:
                self.memory.updateNextState(copy.deepcopy(spiel.board))
            #print(game)
            spiel.switch_player()


	
    def train_step(self,iteration,mode):
        new_batch = []
        taken_colums = []
        #adds all currently saved states to the new Batch
        for idx in range(self.memory.batch_idx):
            flattened_game = self.model.flatten_game(self.model.convert_game(self.memory.mem["state"][idx],self.memory.mem["player"][idx]))
            taken_colums.append(self.model.taken(flattened_game.pop(0)))
            new_batch.append(copy.deepcopy(flattened_game))
        if self.replay.batch_idx > 0:            
            #adds the states with the biggest loss to the new Batch
            for idx in range(self.replay.BATCHSIZE):
                flattened_game = self.model.flatten_game(self.model.convert_game(self.replay.mem["state"][idx],self.replay.mem["player"][idx]))
                taken_colums.append(self.model.taken(flattened_game.pop(0)))
                new_batch.append(copy.deepcopy(flattened_game))

        prediction = self.model(Variable(torch.Tensor(new_batch)))
        target = self.model(Variable(torch.Tensor(new_batch)))
        loss = self.model.loss_func(prediction,target)
        print(loss)
        #updates the states with the biggest loss
        for idx in range(self.memory.batch_idx):
            loss_sub = self.model.loss_func(prediction[idx],target[idx])
            self.replay.addToMem(self.memory.mem["state"],self.memory.mem["next_state"],self.memory.mem["move"],self.memory.mem["player"],self.memory.mem["reward"],loss_sub)
        #makes the predictions of the current states
        for idx in range(self.memory.batch_idx):
            if self.memory.mem["next_state"][idx] is not None:
                flattened_game = self.model.flatten_game(self.model.convert_next_state(self.memory.mem["next_state"][idx],self.memory.mem["player"][idx]))
                free_cols = flattened_game.pop(0)
                target[idx][self.memory.mem["move"][idx]] = -(self.GAMMA * max(self.model(Variable(torch.Tensor(copy.deepcopy(flattened_game)))).tolist()))
                
            else:
                target[idx][self.memory.mem["move"][idx]] = -(self.GAMMA * self.memory.mem["reward"][idx])
            for idx in range(len(free_cols)):
                # give value that probability of move is 0
                move[int(free_cols[idx])] = 0
    
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
                elif mode == 'e':
                    self.epsilon_training()
            self.train_step(self.plotting_step,mode)
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
        self.LR = 0.1
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
    
    def flatten_game(self,game):
        flattened_game = []
        free = ""
        for col in range(COLUMS):
            for row in range(ROWS):
                if game[col][row] == PLAYER1:
                    flattened_game.append(PLAYER1)
                elif game[col][row] == PLAYER2:
                    flattened_game.append(PLAYER2)
                else:
                    flattened_game.append(0)
                    if(free == ""):
                        free = free + chr(col)
                    elif(free[len(free)-1] != col):
                        free = free + chr(col)
        flattened_game.insert(0,free)
        return flattened_game
    
    def convert_game(self,game,player):
    	#return game with empty spaces as 0, the Ki's stones as 1 and the enemys stones as -1
        if player == PLAYER2:
            for col in range(COLUMS):
                for row in range(ROWS):
                    if game[col][row] == PLAYER1:
                        game[col][row] == PLAYER2
                    elif game[col][row] == PLAYER2:
                        game[col][row] == PLAYER1
        return game
    
    def convert_next_state(self,game,player):
    	#return game with empty spaces as 0, the Ki's stones as 1 and the enemys stones as -1
        if player == PLAYER1:
            for col in range(COLUMS):
                for row in range(ROWS):
                    if game[col][row] == PLAYER1:
                        game[col][row] == PLAYER2
                    elif game[col][row] == PLAYER2:
                        game[col][row] == PLAYER1
        return game    
    
    def find_free_cols(self,game):
        free_slots = []
        for col in range(COLUMS):
            free_slots.append(0)
            for row in range(ROWS):
                if game[col][row] == 0:
                    free_slots[col] = 1
                    break
        return free_slots
            
	
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

    def model_move(self,game,taken_cols):
        move = self(game)
        for idx in range(len(taken_cols)):
            # give value that probability of move is 0
            take = taken_cols[idx]
            move[int(take)] = 0
        move = torch.argmax(move).item()
        return move
    
    def taken(self,free_colums):
        taken_colums = "0123456"
        for i in range(len(free_colums)):
            for j in range(len(taken_colums)):
                if taken_colums[j] == free_colums[i]:
                    taken_colums.pop(j)
                    break
        return taken_colums





#EPS_START = 0.1
#EPS_END = 0.98
#EPS_STEPS = 25000





        
training = Trainer()
for i in range(10):
#    training.train(20,'r')
#    training.train(30,'h')
#    training.train(40,'k')
    training.train(5,'e')
    training.model.save_model()
    training.model.LR = training.model.LR / 2

training.writer.close()