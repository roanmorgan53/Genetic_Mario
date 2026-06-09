from genetic_algorithm import GeneticAlgorithm

if __name__ == '__main__':
    GA = GeneticAlgorithm()

    individuals = 20
    steps = 10000
    generations = 1000

    for _ in range(generations):
        GA.run_generation(individuals_per_gen=individuals, max_steps=steps)

    GA.save_best_model(f"best_mario_{generations}g_{individuals}i_{steps}st")
