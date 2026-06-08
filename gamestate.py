
import math
import torch
from typing import Any
import sys

ACTION_SET = [
    [],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['up'],
    ['down'],
    ['A']
]

NUM_GAMESTATE_FEATURES = 8
TILE_SIZE = 16
TILE_GRID_ROWS = 13
TILE_PAGE_WIDTH = 16
TILE_RAM_START = 0x500
PIPE_LOOKAHEAD_TILES = 4
PIPE_MIN_HEIGHT_TILES = 2

class GameState:
    dist_from_start: int
    at_goal: bool
    score: int
    nearest_enemy: int
    time: int
    lives: int
    level: int
    incoming_pipe: bool

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
        self.set_incoming_pipe(env)

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

    def set_incoming_pipe(self, env: Any):
        info = self._get_env_info(env)
        ram = env.unwrapped.ram
        self.incoming_pipe = has_incoming_pipe(ram, int(info["x_pos"]))

    # turns the metadata into a 1D tensor
    def toTensor(self):
        stateList = [
            self.dist_from_start,
            int(self.at_goal),
            self.score,
            self.nearest_enemy,
            self.time,
            self.lives,
            self.level,
            int(self.incoming_pipe)
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

    if nearest_enemy == sys.maxsize:
        return 9999

    return nearest_enemy

def get_level_tile(ram, tile_x, tile_y):
    if tile_y < 0 or tile_y >= TILE_GRID_ROWS:
        return 0

    page = (tile_x // TILE_PAGE_WIDTH) % 2
    sub_x = tile_x % TILE_PAGE_WIDTH
    addr = TILE_RAM_START + page * TILE_GRID_ROWS * TILE_PAGE_WIDTH + tile_y * TILE_PAGE_WIDTH + sub_x

    return int(ram[addr])

def get_column_height(ram, tile_x):
    height = 0

    for tile_y in range(TILE_GRID_ROWS - 1, -1, -1):
        tile = get_level_tile(ram, tile_x, tile_y)

        if tile == 0:
            if height > 0:
                break
            continue

        height += 1

    return height

def has_incoming_pipe(ram, mario_x, lookahead_tiles=PIPE_LOOKAHEAD_TILES):
    mario_tile_x = mario_x // TILE_SIZE

    for tile_offset in range(1, lookahead_tiles + 1):
        left_column_x = mario_tile_x + tile_offset
        right_column_x = left_column_x + 1

        left_height = get_column_height(ram, left_column_x)
        right_height = get_column_height(ram, right_column_x)

        if (
            left_height >= PIPE_MIN_HEIGHT_TILES
            and right_height >= PIPE_MIN_HEIGHT_TILES
            and left_height == right_height
        ):
            return True

    return False

    
