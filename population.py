import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from render import MarioRenderer

POPULATION_SIZE = 3


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

 
def run_generation(envs: list, renderer: MarioRenderer):

    for env in envs:
        env.reset()

        # Lives remaining = 0
        env.unwrapped.ram[0x075A] = 0

    # Array of True for all of population
    is_alive = [True] * POPULATION_SIZE

    # Keep running as long as is_alive has one True
    while any(is_alive):
        
        # If user closes window, stop
        if renderer.window_closed():
            break

        for agent_index in range(POPULATION_SIZE):
            
            # If agent is dead, skip
            if not is_alive[agent_index]:
                continue

            # Access env of current agent
            env = envs[agent_index]

            # If lives == 255, game over
            # lives wrap to 255 after going under 0 since it's an unsigned int
            if env.unwrapped.ram[0x075A] == 255:
                is_alive[agent_index] = False
                continue

            # Pick random action from action space and step
            action = env.action_space.sample()
            
            # step returns: observation, reward, done, info 
            _, _, done, _ = env.step(action)

            # Set to dead
            if done:
                is_alive[agent_index] = False

        
        living_agents = [i for i in range(POPULATION_SIZE) if is_alive[i]]

        # living_agents is empty when all dead
        if not living_agents:
            break

        # unwrapped.ram[0x6D] -> PAGE NUMBER
        # unwrapped.ram[0x86] -> x position within current page
        # Find leading agents by multiplying page # by 256 and adding current x_pos
        # to get an overall x coordinate.
        leader_index = max(
            living_agents,
            key=lambda agent: envs[agent].unwrapped.ram[0x6D] * 256 + envs[agent].unwrapped.ram[0x86],
        )

        renderer.draw(envs, is_alive, leader_index)
        renderer.tick()


def main():
    
    print("Creating " + str({POPULATION_SIZE})  + " environments:")


    envs = [make_env() for _ in range(POPULATION_SIZE)]
    
    # Use MarioRenderer class in render.py
    renderer = MarioRenderer(POPULATION_SIZE)

    
    run_generation(envs, renderer)

    # After generation finished, close renderer and envs
    renderer.close()
    for env in envs:
        env.close()


if __name__ == '__main__':
    main()
