import numpy as np
import gym_super_mario_bros
import torch
import gamestate
import random
from gamestate import GameState
from marionn import MarioNN

# constants
INPUT_SPACE_LENGTH = 9
NUM_ELITES = 2

class GeneticAlgorithm() :
    best = MarioNN()
    modelScores = []
    elites = []
    population = []

    def __init__(self):
        pass

    """
        iterates the genetic algorithm 
    """
    def run_generation(self, env: gym_super_mario_bros.SuperMarioBrosEnv, individuals_per_gen=10, max_steps=1000):
        if not self.population:
            self.generate_solutions(100)
        else:
            self.create_next_generation()

        self.modelScores = []

        # run the models and get the fitness
        for mario in self.population:
            fitness_score = self.fitness(mario, env, max_steps)
            self.modelScores.append((mario, fitness_score))

        

    # create the starting population
    def generate_solutions(self, population_size) -> list[MarioNN]:
        self.population = []

        for _ in range(population_size):
            moustached_italian = MarioNN()
            num_weights = moustached_italian.get_num_weights()

            # populate the weights with random values
            rand_weights = np.random.uniform(-0.05,0.05, size=num_weights)

            # set the weights with the random values
            moustached_italian.set_weights_flat(rand_weights)

            self.population.append(moustached_italian)

        return self.population

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

    """
        given the models and their fitness scores, 
        and the amount of genetic elites, only select the
        best performing models to pass on their genes
    """
    def selection(self):
        self.elites = []
        
        # sort the list by fitness
        sorted = self.modelScores
        sorted.sort(key=lambda x : x[1], reverse=True)

        i = 0
        for model, score in sorted:
            if i >= NUM_ELITES:
                break

            # stop if there are negative weights being pushed
            if score <= 0:
                break

            self.elites.append(model)

            i += 1

        return self.elites

    """
        select individuals from the population tournament style.
    """
    def pick_parent(self, modelScores: list[tuple[MarioNN, float]]):
        if len(modelScores) < 3:
            return MarioNN()

        hunger_games_tributes = []

        # get the three tributes that will fight to continue their bloodline
        while len(hunger_games_tributes) < 3:
            tribute = modelScores[random.randint(0, len(modelScores) - 1)]

            if tribute not in hunger_games_tributes:
                hunger_games_tributes.append(tribute)

        katniss = max(hunger_games_tributes, key=lambda x: x[1])[0]

        return katniss

    """
        generates the new population to run 
    """
    def create_next_generation(self):
        pop_size = len(self.population)
        new_population = []

        # get the elites from the modelScores
        self.selection()

        # keep all the elites as-is
        for elite in self.elites:
            new_population.append(elite)

        remaining = pop_size - len(self.elites)

        # based on parent picking, construct the rest of the population
        for _ in range(remaining):
            mother = self.pick_parent(self.modelScores)
            father = self.pick_parent(self.modelScores)

            child = MarioNN()
            child = child.crossover(father, mother)

            new_population.append(child)
        
        self.population = new_population
        return self.population