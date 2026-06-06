import numpy as np
import gym_super_mario_bros
from gamestate import GameState
from marionn import MarioNN

class GeneticAlgorithm() :
    def __init__(self):
        pass

    # create the starting population
    def generate_solutions(population_size):
        population = []

        for _ in range(population):
            moustached_italian = MarioNN()
            num_weights = moustached_italian.get_num_weights()

            # populate the weights with random values
            rand_weights = np.random.uniform(-0.05,0.05, size=num_weights)

            # set the weights with the random values
            moustached_italian.set_weights_flat(rand_weights)

            population.append(moustached_italian)

        return population

    # sends mario through the level
    # returns a metric of how well the model preformed
    def fitness(model: MarioNN, env: gym_super_mario_bros.SuperMarioBrosEnv, max_steps=10000):

        fitness = 0

        # start from the beginning
        state = env.reset()

        frame = 0
        while frame < max_steps:
            
            # state, reward, done, info = env.step(model.forward())



            frame += 1


        return fitness

    def selection():
        pass