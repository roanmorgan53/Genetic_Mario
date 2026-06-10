import numpy as np
import pygame

# Number of grids (3 by 1) -> 3
GRID_COLS = 3

# Single NES frame
CELL_WIDTH = 256   # one NES frame width
CELL_HEIGHT = 240   # one NES frame height

# Title bar above the grid
TITLE_H = 40

# Canvas is exactly the grid plus the title bar
# Scales to monitor using pygame.SCALED
CANVAS_W = GRID_COLS * CELL_WIDTH
CANVAS_H = CELL_HEIGHT + TITLE_H


# Padding inside each cell, used for making gaps in between cells
CELL_PADDING = 4

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
    Renders 3 mario emulation frames side by side.
    Dead mario frames are grayscale
    """

    def __init__(self, population_size: int):
        self.population_size = population_size

        pygame.init()

        # SCALED stretches canvas to monitor size
        self.screen = pygame.display.set_mode((CANVAS_W, CANVAS_H), pygame.FULLSCREEN | pygame.SCALED)

        pygame.display.set_caption("Genetic Mario")

        self.generation_font = pygame.font.SysFont(None, 36)

        self.quit = False


    # If user closes window, quit the program
    def window_closed(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
        return self.quit

    def draw(self, envs: list, is_alive: list[bool], labels: list[str] = None):
        
        self.screen.fill((0, 0, 0))

        frame_w = CELL_WIDTH - 2 * CELL_PADDING
        frame_h = CELL_HEIGHT - 2 * CELL_PADDING

        # Loop over entire population of agents
        for i in range(self.population_size):
            col = i % GRID_COLS
            row = i // GRID_COLS
            cell_x = col * CELL_WIDTH
            cell_y = TITLE_H + row * CELL_HEIGHT

            # Gets current frame from NES emulator
            frame = envs[i].render(mode='rgb_array')

            # If mario is dead, set frame color(s) to grey
            if not is_alive[i]:
                frame = to_greyscale(frame)

            # frame is: (height, width, 3)
            # pygame needs: (width, height, 3)
            transposed_frame = frame.transpose(1, 0, 2)

            # Scale down and add padding
            surface = pygame.surfarray.make_surface(transposed_frame)
            surface = pygame.transform.scale(surface, (frame_w, frame_h))
            self.screen.blit(surface, (cell_x + CELL_PADDING, cell_y + CELL_PADDING))

            # Render label centered above the frame in the title bar zone
            text = labels[i]
            
            generation_label = self.generation_font.render(text, True, (255, 255, 255))
            label_x = cell_x + (CELL_WIDTH - generation_label.get_width()) // 2
            label_y = (TITLE_H - generation_label.get_height()) // 2
            
            self.screen.blit(generation_label, (label_x, label_y))

        pygame.display.flip()