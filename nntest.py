import gym_super_mario_bros
import nes_py
import torch
import gamestate
from marionn import MarioNN
from genetic_algorithm import GeneticAlgorithm
from nes_py.wrappers import JoypadSpace

GA = GeneticAlgorithm()

pop = GA.generate_solutions(100)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, gamestate.ACTION_SET)

for moustached_italian in pop:
    print(GA.fitness(moustached_italian, env, 1000))
