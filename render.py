import numpy as np
import pygame

# Number of grids (3 by 1) -> 3
GRID_COLS = 3
GRID_ROWS = 1

# Single NES frame
CELL_WIDTH = 256   # one NES frame width
CELL_HEIGHT = 240   # one NES frame height

# Total Grid Dimensions
GRID_W = GRID_COLS * CELL_WIDTH   # 1280
GRID_H = GRID_ROWS * CELL_HEIGHT   # 960

# Title bar above the grid
TITLE_H = 40

# Canvas is exactly the grid plus the title bar
# Scales to monitor using pygame.SCALED
CANVAS_W = GRID_W
CANVAS_H = GRID_H + TITLE_H

OFFSET_X = 0
OFFSET_Y = TITLE_H

# --------------------------------


# The following code is for Java, but works same in python.
# See `intensity` under Method Detail section
# https://introcs.cs.princeton.edu/java/code/javadoc/Luminance.html
def to_greyscale(frame: np.ndarray) -> np.ndarray:
    """
    Converts frame to greyscale by applying the NTSC formula
    """

    # Splice input frame for every r, g, b, vals for every pixel
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]

    # NTSC formula Y = 0.299*r + 0.587*g + 0.114*b
    luminance = (
        0.299 * red +
        0.587 * green +
        0.114 * blue
    ).astype(np.uint8)


    return np.stack([luminance, luminance, luminance], axis=2)


class MarioRenderer:
    """
    Renders 16 Mario agents as a 4×4 grid of NES frames in one pygame window.
    Dead agent windows are greyscale
    """

    def __init__(self, population_size: int):
        self.population_size = population_size

        pygame.init()

        # SCALED stretches canvas to monitor size
        self.screen = pygame.display.set_mode((CANVAS_W, CANVAS_H), pygame.FULLSCREEN | pygame.SCALED)

        pygame.display.set_caption("Genetic Mario")

        self.font       = pygame.font.SysFont(None, 18) # Index Font size = 18
        self.generation_font = pygame.font.SysFont(None, 36) # Generation Font Size = 36

        self.clock      = pygame.time.Clock()
        self.quit       = False

        # Start generation on 1 (duh)
        self.generation = 1


    # If user closes window, quit the program
    def window_closed(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
        return self.quit

    def draw(
        self,
        envs:       list,
        is_alive:   list[bool],
    ):
        
        # Loop over entire population of agents
        for i in range(self.population_size):
            col    = i % GRID_COLS
            row    = i // GRID_COLS
            cell_x = OFFSET_X + col * CELL_WIDTH
            cell_y = OFFSET_Y + row * CELL_HEIGHT

            # Gets current frame from NES emulator
            frame = envs[i].render(mode='rgb_array')
            
            # If mario is dead, set frame color(s) to grey
            if not is_alive[i]:
                frame = to_greyscale(frame)

            # frame is: (height, width, 3)
            # pygame needs: (width, height, 3) 
            transposed_frame = frame.transpose(1, 0, 2)

            # Create surface and display at current cell location
            surface = pygame.surfarray.make_surface(transposed_frame)
            self.screen.blit(surface, (cell_x, cell_y))

            # Render frame # and displays
            number_label = self.font.render("#" + str(i), True, (255, 255, 255))
            self.screen.blit(number_label, (cell_x + 4, cell_y + 4))

        generation_label = self.generation_font.render("Generation " + str(self.generation), True, (255, 255, 255))
        gen_x = (CANVAS_W - generation_label.get_width()) // 2
        gen_y = (OFFSET_Y - generation_label.get_height()) // 2

        self.screen.blit(generation_label, (gen_x, gen_y))

        pygame.display.flip()

    # Cap simulation speed, MAYBE remove?
    def tick(self, fps: int = 0):
        self.clock.tick(fps)

    # Wrapper function for ease of use in population.py
    def set_generation(self, generation: int):
        self.generation = generation

    # Wrapper function so pygame doesn't have to be imported in population.py
    def close(self):
        pygame.quit()