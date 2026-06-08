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
RANDOMS_PER_GENERATION = 2

class GeneticAlgorithm() :
    best_model = MarioNN()
    best_score = -100000
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
            self.generate_solutions(individuals_per_gen)
        else:
            self.create_next_generation()

        self.modelScores = []

        trial_num = 1

        # run the models and get the fitness
        for mario in self.population:
            fitness_score = self.fitness(mario, env, max_steps)
            self.modelScores.append((mario, fitness_score))

            print(f"Trial {trial_num}: fitness {fitness_score}")

            trial_num += 1

            if (fitness_score > self.best_score):
                self.best_score = fitness_score
                self.best_model = mario

                print("Found a new best mario!")
                print(f"New Highscore: {self.best_score}")

    

    # create the starting population
    def generate_solutions(self, population_size) -> list[MarioNN]:
        self.population = self.new_random(population_size) 
        return self.population

    # sends mario through the level
    # returns a metric of how well the model preformed
    def fitness(self, model: MarioNN, env: gym_super_mario_bros.SuperMarioBrosEnv, max_steps=10000):

        fitness = 0
        best_x = 0
        stuck_frames = 0
        prev_score = 0
        prev_coins = 0

        # start from the beginning
        state = env.reset()

        prev_x = -1000

        frame = 0
        while frame < max_steps:
            result_weights = model.forward(GameState(env))

            action = int(torch.argmax(result_weights).item())

            _, _, done, info = env.step(action)

            # baseline value, how far mario went in the x
            x = info["x_pos"]

            if x > best_x:
                fitness += x - best_x   
                best_x = x
                stuck_frames = 0
            else:
                stuck_frames += 1
                fitness -= 0.25

            # score decrease for living too long
            fitness -= 0.05

            fitness += 0.02 * (info["score"] - prev_score)
            fitness += 5 * (info["coins"] - prev_coins)

            # completed the level
            if info["flag_get"]:
                fitness += 1000
                break

            if stuck_frames > 90:
                fitness -= 50
                break

            if done:
                fitness -= 100
                break

            prev_score = info["score"]
            prev_coins = info["coins"]

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
        generates n random marios 
    """
    def new_random(self, num_rand):
        marios = []

        for _ in range(num_rand):
            rmario = MarioNN()
            rand_weights = np.random.uniform(-0.05, 0.05, size=rmario.get_num_weights())
            rmario.set_weights_flat(rand_weights)
            marios.append(rmario)

        return marios


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

        # add the random marios
        for mario in self.new_random(RANDOMS_PER_GENERATION):
            new_population.append(mario)

        remaining = pop_size - RANDOMS_PER_GENERATION - len(self.elites)

        # based on parent picking, construct the rest of the population
        for _ in range(remaining):
            mother = self.pick_parent(self.modelScores)
            father = self.pick_parent(self.modelScores)

            child = MarioNN()
            child = child.crossover(father, mother)
            child.mutate(child, mutation_rate=0.05, mutation_strength=0.1)

            new_population.append(child)
        
        self.population = new_population
        return self.population

    def save_best_model(self, filepath="best_mario_model.pth"):
        if self.best_model is None:
            raise ValueError("No best model is available to save.")

        torch.save(
            {
                "model_state_dict": self.best_model.state_dict(),
                "best_score": self.best_score
            },
            filepath
        )

        return filepath

    def load_model(self, filepath="best_mario_model.pth"):
        checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
        loaded_model = MarioNN()

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            self.best_score = checkpoint.get("best_score", self.best_score)
        else:
            loaded_model.load_state_dict(checkpoint)

        loaded_model.eval()
        self.best_model = loaded_model

        return loaded_model

    def run_model_from_file(
        self,
        filepath,
        env: gym_super_mario_bros.SuperMarioBrosEnv,
        max_steps=1000,
        render=True
    ):
        model = self.load_model(filepath)

        env.reset()
        total_reward = 0.0
        final_info = {}
        steps_taken = 0

        for _ in range(max_steps):
            with torch.no_grad():
                result_weights = model.forward(GameState(env))

            action = int(torch.argmax(result_weights).item())
            _, reward, done, info = env.step(action)

            total_reward += reward
            final_info = info
            steps_taken += 1

            if render:
                env.render()

            if done:
                break

        env.reset()

        return {
            "steps": steps_taken,
            "total_reward": total_reward,
            "info": final_info
        }
