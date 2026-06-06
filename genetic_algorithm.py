import numpy as np
import gym_super_mario_bros
import torch
import gamestate
from gamestate import GameState
from marionn import MarioNN

INPUT_SPACE_LENGTH = 9


class GeneticAlgorithm() :
    def __init__(self):
        pass

    # create the starting population
    def generate_solutions(self, population_size) -> list[MarioNN]:
        population = []

        for _ in range(population_size):
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
    def fitness(self, model: MarioNN, env: gym_super_mario_bros.SuperMarioBrosEnv, max_steps=10000):

        fitness = 0

        # start from the beginning
        state = env.reset()

        frame = 0
        while frame < max_steps:
            result_weights = model.forward(GameState(env))

            action = int(torch.argmax(result_weights).item())

            # print(gamestate.ACTION_SET[action])

            state, reward, done, info = env.step(action)

            # baseline value, how far mario went in the x
            fitness = info["x_pos"]

            # add a percentage of the score
            fitness += 0.1 * info["score"]

            # incentivize coins
            fitness += 2 * info["coins"]

            # disincentivize taking a long time
            fitness -= 0.1 * (400 - info["time"])

            # completed the level
            if info["flag_get"]:
                fitness += 1000

            if done:
                if not info["flag_get"]:
                    fitness -= 100

                break

            env.render()

            frame += 1

        env.reset()

        return fitness

    def selection():
        pass