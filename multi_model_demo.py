import sys
import re
import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gamestate import GameState, ACTION_SET
from marionn import MarioNN
from render import MarioRenderer
from genetic_algorithm import GeneticAlgorithm

NUM_AGENTS = 3


def load_model(filepath: str) -> MarioNN:
    
    # Load state from file
    state = torch.load(filepath, map_location=torch.device("cpu"), weights_only=False)
    
    model = MarioNN()
    
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        raise ValueError("Unrecognized save format! " \
        "Run nntest.py and input its generated files as parameters.")

    model.eval()
    
    return model


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, ACTION_SET)
    return env


def run(models: list[MarioNN], envs: list, renderer: MarioRenderer, labels: list[str] = None):
    for env in envs:
        env.reset()

        # Set lives remaining to 0
        env.unwrapped.ram[0x075A] = 0

    # Initialize all agents to alive
    is_alive = [True] * NUM_AGENTS


    while any(is_alive):
        
        # Check for user closing the window
        if renderer.window_closed():
            break

        # Iterate through agents displayed
        for i in range(NUM_AGENTS):
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
    
    # Arg count check
    if len(sys.argv) != 2:
        print("\nUSAGE: python demo.py 3_model_file")
        sys.exit(1)


    filepaths = sys.argv[1:]
    models = [x[0] for x in GeneticAlgorithm.load_model(self=None, filepath=filepaths[0])]
    
    # Parses "_'X'g_" from filename,
    # using the regular name if "_'X'g_" not found
    labels = [f"Generation {match.group(1)}" 
              if (match := re.search(r'_(\d+)g_', fp)) 
              else fp for fp in filepaths]
    
    labels = labels * 3
    
    envs = [make_env() for _ in range(NUM_AGENTS)]
    renderer = MarioRenderer(NUM_AGENTS)

    run(models, envs, renderer, labels)

    # renderer.close()
    for env in envs:
        env.close()
