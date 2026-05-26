import torch
import torch.nn as nn
import numpy as np

class MarioNN (nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        # TODO: make the nn stack
        # self.stack = 

    # send the game state through the network
    def forward(self, game_state):

        # TODO:
        # 1. turn the game state into a tensor
        # 2. get the logits from the stack
        # 3. ret logits

        pass

    # turn the array into a flat array of weights
    def get_weights_flat(self):
        pass

    # from a 1-Dimensional nparray, set nn weights
    def set_weights_flat(self, weights_flat):
        pass

    def get_num_weights(self):
        return sum(p.numel() for p in self.parameters())

    def create_individual(self):
        model = MarioNN()
        return model

    # make a child given the weights of the parents 
    def crossover(self, parent1, parent2):
        pass

    # randomly change random weights of the model to get non-deterministic behavior 
    def mutate(self, model, mutation_rate=0.1, mutation_strength=0.5):
        pass

    def generate_solutions(population_size):
        pass

    # returns a metric of how well the model preformed
    def fitness():
        pass

    def selection():
        pass