import torch
import torch.nn as nn
import numpy as np
import gamestate
from gamestate import GameState

NUM_GAMESTATE_ELEMENTS = gamestate.NUM_GAMESTATE_FEATURES

# left, right, jump, down, run/fireball, up(vines, optional)
NUM_MOVEMENTS = len(gamestate.ACTION_SET)

# make the number of hidden nodes be between number of inputs and outputs
NUM_HIDDEN_NODES = (NUM_GAMESTATE_ELEMENTS + NUM_MOVEMENTS) // 2

class MarioNN (nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.stack = nn.Sequential(

            # takes in the gamestate
            nn.Linear(NUM_GAMESTATE_ELEMENTS, NUM_HIDDEN_NODES),

            # based on weights and tanh activation fn, gives num -1 <= x <= 1
            nn.Tanh(),

            # outputs the probability of making any specific movements
            nn.Linear(NUM_HIDDEN_NODES, NUM_MOVEMENTS)

        )

    # send the game state through the network
    def forward(self, game_state: GameState) -> torch.Tensor:

        # turn the game state into a tensor
        game_state_tensor = game_state.toTensor()

        # send the game state forward through the network
        outputs = self.stack.forward(game_state_tensor)

        return outputs 

    # turn the array into a flat array of weights
    def get_weights_flat(self):
        weights = []

        for param in self.parameters():
            weights.extend(param.data.flatten().numpy())
        
        # return the weights as a numpy array
        return np.array(weights)

    # from a 1-Dimensional nparray, set nn weights
    def set_weights_flat(self, weights_flat):

        # make sure the weights are the same length
        if len(weights_flat) != self.get_num_weights():
            raise WeightsSizeMismatchError(self.get_num_weights, len(weights_flat))

        i = 0

        for param in self.parameters():
            num_elements = param.numel()
            shape = param.shape

            weights_to_copy = weights_flat[i:(i+num_elements)]

            new_tensor = torch.tensor(weights_to_copy, dtype=param.dtype, device=param.device)

            new_tensor = new_tensor.view(shape)

            with torch.no_grad():
                param.copy_(new_tensor)

            i += num_elements

        if i != len(weights_flat):
            print(f"i might be off: i = {i}, weights len = {len(weights_flat)}")


    def get_num_weights(self):
        return sum(p.numel() for p in self.parameters())

    def create_individual():
        model = MarioNN()
        return model

    # make a child given the weights of the parents 
    def crossover(self, parent1, parent2):
        child = MarioNN()

        father_flat = parent1.get_weights_flat()
        mother_flat = parent2.get_weights_flat()

        # get a random point for gene mixing
        crossover_point = np.random.randint(len(father_flat))

        child_weights = np.concatenate([
            father_flat[:crossover_point],
            mother_flat[crossover_point:]
        ])

        child.set_weights_flat(child_weights)
        return child

    # randomly change random weights of the model to get non-deterministic behavior 
    def mutate(self, model, mutation_rate=0.1, mutation_strength=0.5):
        weights = model.get_weights_flat()

        # randomly choose the weights to mutate
        mask = np.random.random(len(weights)) < mutation_rate

        # apply the mutation on the weights
        weights[mask] += np.random.randn(mask.sum()) * mutation_strength

        model.set_weights_flat(weights)

        return model

class WeightsSizeMismatchError(Exception):
    def __init__(self, expected, received):
        super().__init__(f"Expected {expected}, received {received}")
