from genetic_algorithm import GeneticAlgorithm
import time

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':
    GA = GeneticAlgorithm()

    individuals = 100
    steps = 10000
    generations = 4000

    for _ in range(generations):
        # start = time.perf_counter()

        GA.run_generation(individuals_per_gen=individuals, max_steps=steps)

        # end = time.perf_counter()
        # print(f"generation {_} runtime: {end - start}")

    GA.save_best_model(f"best_mario_{generations}g_{individuals}i_{steps}st.pth")

 
