from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gamestate
import time
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, [['right']])


env.reset()

env.unwrapped.ram[0x075A] = 0

done = True


for step in range(50000):


    EXTRA_LIVES = env.unwrapped.ram[0x075A]

    # Check if mario has died
    # Extra lives wraps back around to 255, since its an unsigned int
    if EXTRA_LIVES == 255:
        print("MARIO DEAD")
        env.close()
        break

    state, reward, done, info = env.step(env.action_space.sample())

    ram = env.unwrapped.ram

    # print(gamestate.get_nearest_enemy(gamestate.get_enemy_positions(ram), info["x_pos"], info["y_pos"]))

    env.render()