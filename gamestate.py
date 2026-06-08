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

NUM_GAMESTATE_FEATURES = 11
TILE_SIZE = 16
TILE_GRID_ROWS = 13
TILE_PAGE_WIDTH = 16
TILE_RAM_START = 0x500
LOOKAHEAD_TILES = 6
MAX_STEP_DOWN_TILES = 1
SMALL_BODY_HEIGHT_TILES = 2
BIG_BODY_HEIGHT_TILES = 3
NO_FEATURE_DISTANCE = LOOKAHEAD_TILES + 1

class GameState:
    dist_from_start: int
    at_goal: bool
    score: int
    nearest_enemy: int
    time: int
    lives: int
    level: int
    obstacle_ahead: bool
    obstacle_distance: int
    hole_ahead: bool
    hole_distance: int

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
        self.set_terrain_features(env)

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

    def set_terrain_features(self, env: Any):
        info = self._get_env_info(env)
        ram = env.unwrapped.ram
        mario_x = int(info["x_pos"])
        mario_y = int(info["y_pos"])
        status = str(info["status"])

        self.obstacle_distance = find_next_obstacle_distance(ram, mario_x, mario_y, status)
        self.obstacle_ahead = self.obstacle_distance <= LOOKAHEAD_TILES

        self.hole_distance = find_next_hole_distance(ram, mario_x, mario_y)
        self.hole_ahead = self.hole_distance <= LOOKAHEAD_TILES

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
            int(self.obstacle_ahead),
            self.obstacle_distance,
            int(self.hole_ahead),
            self.hole_distance
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

def is_solid_tile(tile):
    return tile != 0

def get_level_tile(ram, tile_x, tile_y):
    if tile_y < 0 or tile_y >= TILE_GRID_ROWS:
        return 0

    page = (tile_x // TILE_PAGE_WIDTH) % 2
    sub_x = tile_x % TILE_PAGE_WIDTH
    addr = TILE_RAM_START + page * TILE_GRID_ROWS * TILE_PAGE_WIDTH + tile_y * TILE_PAGE_WIDTH + sub_x

    return int(ram[addr])

def y_pos_to_tile_row(y_pos):
    pixels_from_bottom = max(int(y_pos) - 1, 0)
    tile_row = TILE_GRID_ROWS - 1 - (pixels_from_bottom // TILE_SIZE)

    return max(0, min(TILE_GRID_ROWS - 1, tile_row))

def get_body_height_tiles(status):
    if status in ("tall", "fireball"):
        return BIG_BODY_HEIGHT_TILES

    return SMALL_BODY_HEIGHT_TILES

def find_support_row(ram, mario_tile_x, mario_row_guess):
    start_row = max(0, min(TILE_GRID_ROWS - 1, mario_row_guess))

    for tile_row in range(start_row, TILE_GRID_ROWS):
        for tile_x in (mario_tile_x, mario_tile_x + 1):
            if is_solid_tile(get_level_tile(ram, tile_x, tile_row)):
                return tile_row

    return None

def column_has_support(ram, tile_x, support_row, max_step_down_tiles=MAX_STEP_DOWN_TILES):
    if support_row is None:
        return False

    deepest_row = min(TILE_GRID_ROWS - 1, support_row + max_step_down_tiles)

    for tile_row in range(support_row, deepest_row + 1):
        if is_solid_tile(get_level_tile(ram, tile_x, tile_row)):
            return True

    return False

def footprint_has_support(ram, left_tile_x, support_row, max_step_down_tiles=MAX_STEP_DOWN_TILES):
    for tile_x in (left_tile_x, left_tile_x + 1):
        if column_has_support(ram, tile_x, support_row, max_step_down_tiles):
            return True

    return False

def column_has_body_obstacle(ram, tile_x, support_row, body_height_tiles):
    if support_row is None:
        return False

    top_row = max(0, support_row - body_height_tiles)
    bottom_row = max(0, support_row - 1)

    for tile_row in range(top_row, bottom_row + 1):
        if is_solid_tile(get_level_tile(ram, tile_x, tile_row)):
            return True

    return False

def body_path_has_obstacle(ram, left_tile_x, support_row, body_height_tiles):
    for tile_x in (left_tile_x, left_tile_x + 1):
        if column_has_body_obstacle(ram, tile_x, support_row, body_height_tiles):
            return True

    return False

def find_next_hole_distance(ram, mario_x, mario_y, lookahead_tiles=LOOKAHEAD_TILES):
    mario_tile_x = mario_x // TILE_SIZE
    mario_row_guess = y_pos_to_tile_row(mario_y)
    support_row = find_support_row(ram, mario_tile_x, mario_row_guess)

    if support_row is None:
        return NO_FEATURE_DISTANCE

    for tile_offset in range(1, lookahead_tiles + 1):
        left_tile_x = mario_tile_x + tile_offset

        if not footprint_has_support(ram, left_tile_x, support_row):
            return tile_offset

    return NO_FEATURE_DISTANCE

def find_next_obstacle_distance(ram, mario_x, mario_y, status, lookahead_tiles=LOOKAHEAD_TILES):
    mario_tile_x = mario_x // TILE_SIZE
    mario_row_guess = y_pos_to_tile_row(mario_y)
    support_row = find_support_row(ram, mario_tile_x, mario_row_guess)

    if support_row is None:
        return NO_FEATURE_DISTANCE

    body_height_tiles = get_body_height_tiles(status)

    for tile_offset in range(1, lookahead_tiles + 1):
        left_tile_x = mario_tile_x + tile_offset

        if body_path_has_obstacle(ram, left_tile_x, support_row, body_height_tiles):
            return tile_offset

    return NO_FEATURE_DISTANCE

    
