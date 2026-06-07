import gym_super_mario_bros
import nes_py
import torch
import gamestate
import genetic_algorithm
from marionn import MarioNN
from genetic_algorithm import GeneticAlgorithm
from nes_py.wrappers import JoypadSpace

GA = GeneticAlgorithm()

pop = GA.generate_solutions(50)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, gamestate.ACTION_SET)

original_reset = env.reset

def reset_with_one_life(*args, **kwargs):
    state = original_reset(*args, **kwargs)
    env.unwrapped.ram[0x075A] = 0
    return state

env.reset = reset_with_one_life
env.reset()

for _ in range(50): 
    GA.run_generation(env, individuals_per_gen=10, max_steps=500)
