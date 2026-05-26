from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gamestate
import time
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())

    ram = env.unwrapped.ram

    print(gamestate.get_nearest_enemy(gamestate.get_enemy_positions(ram), info["x_pos"], info["y_pos"]))
    env.render()

env.close()