# Genetic Mario

Mario teaches himself how to play 1-1 using a genetic algorithm.

**Authors:** Roan Morgan, Hayk Chaloyan, Jonathan Martin

## Project Overview

**Neural network:** `MarioNN` 
- Input: 11 game state features.

- Output: A score for each of the 11 possible actions.

- Highest Scoring action is taken every frame.

![Neural Net Diagram](diagrams/Neural%20Net%20Diagram.jpg)


**Game state inputs:** [State read from NES RAM]
1. Distance From the Start
2. Is Mario at the goal
3. Score
4. Nearest Enemy
5. Time
6. Lives
7. Level 
8. Is there an obstacle ahead of Mario?
9. Mario’s distance to that obstacle
10. Is there a pit of doom in front of Mario? 
11. Mario’s distance to the pit of doom


**Fitness Calculation:** [Runs every frame]
- Forward progress in the x
- Level completion
- Getting stuck
- Dying
- Mario score
- Coins


**Genetic algorithm:**
- Evaluate the whole population in parallel (one process per CPU core)

- Keep the top `NUM_ELITES` models unchanged

- Fill the rest of the next generation via:
  - **Tournament selection:** Randomly sample a few individuals, the highest scorer becomes a parent.

  - **Single-point crossover:** Take parent 1's genome weights, parent 2's genome weights. To determine the child's genes, randomly select a "crossover point, which determines which genes are from which parent. See diagram below.

    ![Crossover Diagram](diagrams/Crossover%20Diagram.jpg)

  - **Random weight mutation:** Each weight has a small chance of being nudged by a random amount, keeping the population from converging to a suboptimal strategy.
  
- Inject a small number of fresh random individuals to maintain diversity

## Setup

```bash
pip install -r requirements.txt
```

**Tested on Python 3.11.** Key version constraints:
- `gym==0.25.1` — not compatible with gym 0.26+
- `numpy==1.26.4` — not compatible with numpy 2.0+
- `nes-py==8.2.1`
- `gym-super-mario-bros==7.4.0`

## Training

```bash
python train.py
```

| Argument | Default | Description |
|---|---|---|
| `--generations` | 2000 | Number of generations to run |
| `--population` | 40 | Individuals per generation |
| `--steps` | 10000 | Max steps per individual per generation |

Example:
```bash
python train.py --generations 100 --population 40 --steps 10000
```

After training, the top 3 models are saved as `.pth` files:
```
best_mario_{generations}g_{population}i_{steps}st_rank1.pth
best_mario_{generations}g_{population}i_{steps}st_rank2.pth
best_mario_{generations}g_{population}i_{steps}st_rank3.pth
```

NOTE: You can configure the number of models saved
Simply update MAX_BEST_MODELS in config.py

## Demo

Render 1–3 trained models side by side:

```bash
python demo.py model.pth
python demo.py model1.pth model2.pth
python demo.py model1.pth model2.pth model3.pth
```

Live models render in color. Dead models are paused and grayscale. 
Generation labels are parsed from the filename
- EX: `_100g_` renders as **Generation 100**.

## Configuration

All constant parameters are in `config.py`:

```python
# Genetic algorithm constants / parameters
NUM_ELITES = 4
RANDOMS_PER_GENERATION = 4
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5
INITIAL_WEIGHT_RANGE = (-0.05, 0.05)

# Fitness scoring
STUCK_FRAME_THRESHOLD = 90
STUCK_FRAME_PENALTY = 0.25
TIME_PENALTY_PER_FRAME = 0.05
GAME_SCORE_WEIGHT = 0.02
COIN_BONUS = 5
LEVEL_COMPLETION_BONUS = 1000
STUCK_BREAK_PENALTY = 50
DEATH_PENALTY = 100

# How many models get saved at the end of train.py
MAX_BEST_MODELS = 3
```
