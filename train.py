import argparse
import warnings
from genetic_algorithm import GeneticAlgorithm


# Run python train.py --help

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the genetic algorithm." \
    " Running with no arguments runs with default values.")
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    GA = GeneticAlgorithm()

    for gen in range(1, args.generations + 1):
        print(f"Generation {gen}/{args.generations}")
        GA.run_generation(individuals_per_gen=args.population, max_steps=args.steps)

    # base filename that we build on
    base = f"best_mario_{args.generations}g_{args.population}i_{args.steps}st"
    
    # Iterate through the top models
    for rank, (model, score) in enumerate(reversed(GA.best_models), start=1):
        
        # Add rank to base filename
        filepath = f"{base}_rank{rank}.pth"
        
        GeneticAlgorithm.save_model(model, score, filepath)
        
        print(f"Saved rank {rank} model (score {score:.2f}) to {filepath}")
