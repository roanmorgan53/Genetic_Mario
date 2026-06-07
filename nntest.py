import gym_super_mario_bros
import nes_py
import torch
import gamestate
from marionn import MarioNN
from genetic_algorithm import GeneticAlgorithm
from nes_py.wrappers import JoypadSpace

GA = GeneticAlgorithm()

pop = GA.generate_solutions(50)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, gamestate.ACTION_SET)

modelScores = [] 

for moustached_italian in pop:
    fit = GA.fitness(moustached_italian, env, 100)

    modelScores.append((moustached_italian, fit))
     
GA.selection(modelScores, 10)
GA.pick_parent(modelScores)
