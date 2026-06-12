import sys
import os
import warnings
import re
import torch

stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

import gym_super_mario_bros

sys.stderr.close()
sys.stderr = stderr

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from nes_py.wrappers import JoypadSpace
from gamestate import GameState, ACTION_SET
from marionn import MarioNN
from render import MarioRenderer
from genetic_algorithm import GeneticAlgorithm


def load_model(filepath: str) -> MarioNN:
    # Load a model from .pth export of GeneticAlgorithm
    model, _ = GeneticAlgorithm.load_model(filepath)
    model.eval()
    return model


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, ACTION_SET)
    return env


def run(models: list[MarioNN], envs: list, renderer: MarioRenderer, labels: list[str] = None):
    num_agents = len(models)

    for env in envs:
        env.reset()

        # Set lives remaining to 0
        env.unwrapped.ram[0x075A] = 0

    is_alive = [True] * num_agents

    while any(is_alive):

        # Check for user closing the window
        if renderer.window_closed():
            break

        # Iterate through agents displayed
        for i in range(num_agents):
            if not is_alive[i]:
                continue

            env = envs[i]

            # lives wrap to 255 after going under 0 (unsigned int)
            if env.unwrapped.ram[0x075A] == 255:
                is_alive[i] = False
                continue

            with torch.no_grad():
                action = int(torch.argmax(models[i].forward(GameState(env))).item())

            _, _, done, _ = env.step(action)

            if done:
                is_alive[i] = False

        renderer.draw(envs, is_alive, labels)


if __name__ == '__main__':

    if not (1 <= len(sys.argv) - 1 <= 3):
        print("""
        USAGE:
            
            python demo.py [model1]
            python demo.py [model1] [model2]
            python demo.py [model1] [model2] [model3]
              
        demo.py supports the rendering of 1-3 environments.
            """)
        sys.exit(1)

    filepaths = sys.argv[1:]
    num_agents = len(filepaths)
    models = [load_model(fp) for fp in filepaths]

    # Parses "_'X'g_" from filename,
    # using the regular name if "_'X'g_" not found
    labels = [f"Generation {match.group(1)}"
              if (match := re.search(r'_(\d+)g_', fp))
              else fp for fp in filepaths]

    envs = [make_env() for _ in range(num_agents)]
    renderer = MarioRenderer(num_agents)

    run(models, envs, renderer, labels)

    renderer.close()
    for env in envs:
        env.close()
