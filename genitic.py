import time
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
POPULATION_SIZE = 2048
MAX_GENERATIONS = 1000
ELITISM_RATE = 0.10
MUTATION_RATE = 0.25
TARGET_STRING = "Hello world!"
NO_IMPROVEMENT_LIMIT = 100

# User input
CROSSOVER_TYPE = input("Enter crossover type (SINGLE / TWO / UNIFORM): ").strip().upper()
FITNESS_MODE = input("Choose fitness type (ascii / lcs / combined): ").strip().lower()

# Longest Common Subsequence
def lcs_length(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]

# Individual
class Individual:
    def __init__(self, value=None):
        self.genes = value or self._generate_random_genes()
        self.fitness = 0

    @staticmethod
    def _generate_random_genes():
        return ''.join(random.choice(string.printable[:95]) for _ in range(len(TARGET_STRING)))

# Population functions
def create_population():
    return [Individual() for _ in range(POPULATION_SIZE)]

def evaluate_population(population):
    for individual in population:
        if FITNESS_MODE == "ascii":
            individual.fitness = sum(abs(ord(individual.genes[i]) - ord(TARGET_STRING[i])) for i in range(len(TARGET_STRING)))
        elif FITNESS_MODE == "lcs":
            individual.fitness = -lcs_length(individual.genes, TARGET_STRING)
        elif FITNESS_MODE == "combined":
            ascii_distance = sum(abs(ord(individual.genes[i]) - ord(TARGET_STRING[i])) for i in range(len(TARGET_STRING)))
            lcs_score = lcs_length(individual.genes, TARGET_STRING)
            individual.fitness = ascii_distance - (lcs_score * 2)
        else:
            raise ValueError("Invalid FITNESS_MODE. Choose ascii, lcs, or combined.")

def sort_population(population):
    population.sort(key=lambda x: x.fitness)

def apply_elitism(current_population, next_generation, elite_size):
    for i in range(elite_size):
        next_generation[i].genes = current_population[i].genes
        next_generation[i].fitness = current_population[i].fitness

def mutate(individual):
    index = random.randint(0, len(TARGET_STRING) - 1)
    genes = list(individual.genes)
    genes[index] = random.choice(string.printable[:95])
    individual.genes = ''.join(genes)

def crossover(current_population, next_generation):
    elite_size = int(POPULATION_SIZE * ELITISM_RATE)
    apply_elitism(current_population, next_generation, elite_size)

    for i in range(elite_size, POPULATION_SIZE):
        parent1 = random.randint(0, POPULATION_SIZE // 2 - 1)
        parent2 = random.randint(0, POPULATION_SIZE // 2 - 1)
        genes1 = current_population[parent1].genes
        genes2 = current_population[parent2].genes

        if CROSSOVER_TYPE == "SINGLE":
            point = random.randint(1, len(TARGET_STRING) - 1)
            child_genes = genes1[:point] + genes2[point:]
        elif CROSSOVER_TYPE == "TWO":
            p1 = random.randint(0, len(TARGET_STRING) - 2)
            p2 = random.randint(p1 + 1, len(TARGET_STRING) - 1)
            child_genes = genes1[:p1] + genes2[p1:p2] + genes1[p2:]
        elif CROSSOVER_TYPE == "UNIFORM":
            child_genes = ''.join(random.choice([g1, g2]) for g1, g2 in zip(genes1, genes2))
        else:
            raise ValueError("Invalid crossover type. Choose SINGLE, TWO, or UNIFORM.")

        child = Individual(child_genes)
        if random.random() < MUTATION_RATE:
            mutate(child)

        next_generation[i] = child

# Main algorithm
def run_genetic_algorithm():
    current_generation = create_population()
    next_generation = [Individual() for _ in range(POPULATION_SIZE)]

    stats_log = []
    fitness_log = []

    best_fitness = None
    no_improvement_counter = 0

    for gen in range(MAX_GENERATIONS):
        start_time = time.time()

        evaluate_population(current_generation)
        sort_population(current_generation)

        fitness_values = [ind.fitness for ind in current_generation]
        fitness_log.append(fitness_values.copy())

        best = fitness_values[0]
        worst = fitness_values[-1]
        avg = np.mean(fitness_values)
        std_dev = np.std(fitness_values)
        elapsed_time = time.time() - start_time

        stats_log.append({
            "Generation": gen,
            "Best Fitness": best,
            "Worst Fitness": worst,
            "Average Fitness": avg,
            "Std Dev": std_dev,
            "Elapsed Time (s)": elapsed_time
        })

        print(f"Generation {gen}: Best={best}, Avg={avg:.2f}, Worst={worst}, StdDev={std_dev:.2f}")
        print(f"Best match: {current_generation[0].genes}")

        # ✅ STOP IF EXACT MATCH FOUND
        if current_generation[0].genes == TARGET_STRING:
            print("Target string reached! Stopping early.")
            break

        # ✅ Stop if no improvement in N generations
        if best_fitness is None or best < best_fitness:
            best_fitness = best
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= NO_IMPROVEMENT_LIMIT:
                print(f"No improvement in {NO_IMPROVEMENT_LIMIT} generations. Stopping early.")
                break

        crossover(current_generation, next_generation)
        current_generation, next_generation = next_generation, current_generation

    return pd.DataFrame(stats_log), fitness_log

# Run and plot results
results_df, fitness_over_time = run_genetic_algorithm()

plt.figure(figsize=(12, 6))
plt.plot(results_df['Generation'], results_df['Best Fitness'], label='Best Fitness', linewidth=2)
plt.plot(results_df['Generation'], results_df['Average Fitness'], label='Average Fitness', linestyle='--')
plt.plot(results_df['Generation'], results_df['Worst Fitness'], label='Worst Fitness', linestyle=':')
plt.title('Fitness Progression Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.boxplot(fitness_over_time, vert=True, patch_artist=True, showfliers=True)
plt.title('Fitness Distribution Per Generation (Boxplot)')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.grid(True)
plt.tight_layout()
plt.show()
