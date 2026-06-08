import gym_super_mario_bros
import nes_py
import torch
import gamestate
import genetic_algorithm
from marionn import MarioNN
from genetic_algorithm import GeneticAlgorithm
from nes_py.wrappers import JoypadSpace

GA = GeneticAlgorithm()

env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, gamestate.ACTION_SET)

original_reset = env.reset

def reset_with_one_life(*args, **kwargs):
    state = original_reset(*args, **kwargs)
    env.unwrapped.ram[0x075A] = 0
    return state

env.reset = reset_with_one_life
env.reset()

individuals = 20
steps = 10000
generations = 1000 

for _ in range(generations): 
    GA.run_generation(env, individuals_per_gen=individuals, max_steps=steps)

GA.save_best_model(f"best_mario_{generations}g_{individuals}i_{steps}st")
