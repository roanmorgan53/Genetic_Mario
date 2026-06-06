
import math
import torch
from typing import Any

class GameState:
    dist_from_start: int
    at_goal: bool
    score: int
    nearest_enemy: int
    time: int
    lives: int
    level: int

    def __init__(self, env: Any):
        self.refresh_from_env(env)

    def refresh_from_env(self, env: Any):
        self.set_dist_from_start(env)
        self.set_at_goal(env)
        self.set_score(env)
        self.set_nearest_enemy(env)
        self.set_time(env)
        self.set_lives(env)
        self.set_level(env)

    def _get_env_info(self, env: Any):
        return env.unwrapped._get_info()

    def set_dist_from_start(self, env: Any):
        info = self._get_env_info(env)
        self.dist_from_start = int(info["x_pos"])

    def set_at_goal(self, env: Any):
        info = self._get_env_info(env)
        self.at_goal = bool(info["flag_get"])

    def set_score(self, env: Any):
        info = self._get_env_info(env)
        self.score = int(info["score"])

    def set_nearest_enemy(self, env: Any):
        info = self._get_env_info(env)
        ram = env.unwrapped.ram
        enemy_positions = get_enemy_positions(ram)
        self.nearest_enemy = get_nearest_enemy(
            enemy_positions,
            int(info["x_pos"]),
            int(info["y_pos"])
        )

    def set_time(self, env: Any):
        info = self._get_env_info(env)
        self.time = int(info["time"])

    def set_lives(self, env: Any):
        info = self._get_env_info(env)
        lives = int(info["life"])
        self.lives = 0 if lives == 0xFF else lives

    def set_level(self, env: Any):
        info = self._get_env_info(env)
        world = int(info["world"])
        stage = int(info["stage"])
        self.level = world * 10 + stage

    # turns the metadata into a 1D tensor
    def toTensor(self):
        stateList = [
            self.dist_from_start,
            int(self.at_goal),
            self.score,
            self.nearest_enemy,
            self.time,
            self.lives,
            self.level
        ]

        t = torch.tensor(stateList, dtype=torch.float32)

        return t

def get_enemy_positions(ram):
    enemies = []

    for i in range(5):
        enemy_drawn = ram[0x0F + i]

        if enemy_drawn:
            enemy_x = ram[0x6E + i] * 0x100 + ram[0x87 + i]
            enemy_y = ram[0xCF + i]

            enemies.append((enemy_x, enemy_y))

    return enemies

def get_nearest_enemy(enemy_positions, mario_x, mario_y):

    # default to the largest int val possible 
    nearest_enemy = sys.maxsize

    for enemy_x, enemy_y in enemy_positions:
        man_dist = abs(mario_x - enemy_x) + abs(mario_y - enemy_y)

        if man_dist < nearest_enemy:
            nearest_enemy = man_dist

    if math.isinf(nearest_enemy):
        return 9999

    return nearest_enemy

    
