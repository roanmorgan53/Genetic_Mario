
import time
import math
import torch

class GameState:
    dist_from_start: int
    at_goal: bool
    score: int
    nearest_enemy: int
    time: int
    lives: int
    level: int

    def __init__(
            self, 
            dist_from_start: int, 
            at_goal: bool, 
            nearest_enemy: int,
            score: int, 
            time:int, 
            lives: int = 3, 
            level:int = 1
        ):
                       
        self.dist_from_start = dist_from_start
        self.at_goal = at_goal
        self.nearest_enemy = nearest_enemy
        self.score = score
        self.time = time
        self.lives = lives
        self.level = level

    # turns the metadata into a 1D tensor
    def toTensor(self):
        stateList = [
            self.dist_from_start,
            self.at_goal,
            self.score,
            self.nearest_enemy,
            self.time,
            self.lives,
            self.level
        ]

        t = torch.tensor(stateList)
        print ("Gamestate Tensor:", t)

        return t

def get_enemy_positions(ram):
    enemies = []

    for i in range(5):
        enemy_drawn = ram[0x0F + i]

        if enemy_drawn:
            enemy_x = ram[0x6E + i] * 0x100 + ram[0x87 + i]
            enemy_y = ram[0xCF + i]

            enemies.append((enemy_x, enemy_y))

    
    if len(enemies) > 0:
        print(enemies)
    

    return enemies

def get_nearest_enemy(enemy_positions, mario_x, mario_y):

    nearest_enemy = math.inf
    for enemy_x, enemy_y in enemy_positions:
        man_dist = abs(mario_x - enemy_x) + abs(mario_y - enemy_y)

        if man_dist < nearest_enemy:
            nearest_enemy = man_dist

    if(nearest_enemy <= 70):
        print(nearest_enemy)
        print((mario_x, mario_y))
        print((enemy_x, enemy_y))

    return nearest_enemy

    