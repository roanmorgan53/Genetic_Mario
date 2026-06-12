import multiprocessing
import numpy as np
import os
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

import gym_super_mario_bros

sys.stderr.close()
sys.stderr = stderr

import torch
import gamestate
import random
from nes_py.wrappers import JoypadSpace
from gamestate import GameState
from marionn import MarioNN
from config import (
    NUM_ELITES,
    RANDOMS_PER_GENERATION,
    MAX_BEST_MODELS,
    TOURNAMENT_SIZE,
    MUTATION_RATE,
    MUTATION_STRENGTH,
    INITIAL_WEIGHT_RANGE,
    STUCK_FRAME_THRESHOLD,
    STUCK_FRAME_PENALTY,
    TIME_PENALTY_PER_FRAME,
    GAME_SCORE_WEIGHT,
    COIN_BONUS,
    LEVEL_COMPLETION_BONUS,
    STUCK_BREAK_PENALTY,
    DEATH_PENALTY,
    BACKWARD_PENALTY,
)


def fitness(model: MarioNN, max_steps=10000):
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, gamestate.ACTION_SET)

    score = 0
    best_x = 0
    prev_x = 0
    stuck_frames = 0
    prev_score = 0
    prev_coins = 0

    env.reset()
    env.unwrapped.ram[0x075A] = 0

    frame = 0
    while frame < max_steps:
        result_weights = model.forward(GameState(env))

        action = int(torch.argmax(result_weights).item())

        _, _, done, info = env.step(action)

        # baseline value, how far mario went in the x
        x = info["x_pos"]

        if x > best_x:
            score += x - best_x
            best_x = x
            stuck_frames = 0
        else:
            stuck_frames += 1
            score -= STUCK_FRAME_PENALTY

        if x < prev_x:
            score -= (prev_x - x) * BACKWARD_PENALTY

        prev_x = x

        # score decrease for living too long
        score -= TIME_PENALTY_PER_FRAME

        score += GAME_SCORE_WEIGHT * (info["score"] - prev_score)
        score += COIN_BONUS * (info["coins"] - prev_coins)

        # completed the level
        if info["flag_get"]:
            score += LEVEL_COMPLETION_BONUS
            break

        if stuck_frames > STUCK_FRAME_THRESHOLD:
            score -= STUCK_BREAK_PENALTY
            break

        if done:
            score -= DEATH_PENALTY
            break

        prev_score = info["score"]
        prev_coins = info["coins"]

        frame += 1

    env.reset()
    env.close()

    return score


# constants
INPUT_SPACE_LENGTH = gamestate.NUM_GAMESTATE_FEATURES

class GeneticAlgorithm():

    def __init__(self):
        self.best_models = []
        self.best_score = float('-inf')
        self.model_scores = []
        self.elites = []
        self.population = []

    """
        iterates the genetic algorithm
    """
    def run_generation(self, individuals_per_gen=10, max_steps=1000):
        if not self.population:
            self.generate_solutions(individuals_per_gen)
        else:
            self.create_next_generation()

        self.model_scores = []

        num_workers = min(multiprocessing.cpu_count(), len(self.population))

        # run the models and get the fitness
        with multiprocessing.Pool(num_workers) as pool:
            scores = pool.starmap(fitness, [(mario, max_steps) for mario in self.population])

        for trial_num, (mario, fitness_score) in enumerate(zip(self.population, scores), start=1):
            self.model_scores.append((mario, fitness_score))

            # print(f"Trial {trial_num}: fitness {fitness_score}")

            if (fitness_score > self.best_score):
                self.best_score = fitness_score

                if(len(self.best_models) == MAX_BEST_MODELS):
                    self.best_models.pop(0)
                self.best_models.append((mario, self.best_score))


                # print("Found a new best mario!")
                print(f"New Highscore: {self.best_score}")



    # create the starting population
    def generate_solutions(self, population_size) -> list[MarioNN]:
        self.population = self.new_random(population_size)
        return self.population

    """
        given the models and their fitness scores,
        and the amount of genetic elites, only select the
        best performing models to pass on their genes
    """
    def selection(self):
        self.elites = []

        # sort the list by fitness
        ranked = self.model_scores
        ranked.sort(key=lambda x : x[1], reverse=True)

        i = 0
        for model, score in ranked:
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
        if len(modelScores) < TOURNAMENT_SIZE:
            return MarioNN()

        hunger_games_tributes = []

        # get the three tributes that will fight to continue their bloodline
        while len(hunger_games_tributes) < TOURNAMENT_SIZE:
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

        lo, hi = INITIAL_WEIGHT_RANGE
        for _ in range(num_rand):
            rmario = MarioNN()
            rand_weights = np.random.uniform(lo, hi, size=rmario.get_num_weights())
            rmario.set_weights_flat(rand_weights)
            marios.append(rmario)

        return marios


    """
        generates the new population to run
    """
    def create_next_generation(self):
        pop_size = len(self.population)
        new_population = []

        # get the elites from the model_scores
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
            mother = self.pick_parent(self.model_scores)
            father = self.pick_parent(self.model_scores)

            child = MarioNN()
            child = child.crossover(father, mother)
            child.mutate(mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH)

            new_population.append(child)

        self.population = new_population
        return self.population

    @staticmethod
    def save_model(model: MarioNN, score: float, filepath: str):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_score": float(score)
            },
            filepath
        )

    def save_best_model(self, filepath="best_mario_model.pth"):
        if not self.best_models:
            raise ValueError("No best model is available to save.")

        # save only the single best model found so far
        model, score = self.best_models[-1]
        GeneticAlgorithm.save_model(model, score, filepath)

        return filepath

    @staticmethod
    def load_model(filepath="best_mario_model.pth"):
        checkpoint = torch.load(filepath, weights_only=True)

        model = MarioNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        score = checkpoint.get("best_score", None)

        return model, score

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
