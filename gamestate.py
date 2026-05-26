

class GameState:
    dist_from_goal: int
    at_goal: bool
    score: int
    time: int
    lives: int
    level: int

    def __init__(self, dist_from_goal: int, 
                       at_goal: bool, 
                       score: int, 
                       time:int, 
                       lives: int = 3, 
                       level:int = 1):
        self.dist_from_goal = dist_from_goal
        self.at_goal = at_goal
        self.score = score
        self.time = time
        self.lives = lives
        self.level = level


def position_to_goal(x: int, y: int):
    pass