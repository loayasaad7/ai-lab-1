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
TOURNAMENT_K = 5
STOCHASTIC_P = 0.8
AGING_MAX_AGE = 10

# User input
try:
    CROSSOVER_TYPE = input("Enter crossover type (SINGLE / TWO / UNIFORM): ").strip().upper()
    assert CROSSOVER_TYPE in {"SINGLE", "TWO", "UNIFORM"}

    FITNESS_MODE = input("Choose fitness type (ascii / lcs / combined): ").strip().lower()
    assert FITNESS_MODE in {"ascii", "lcs", "combined"}

    SELECTION_METHOD = input("Enter parent selection method (RWS / SUS / TOURNAMENT / STOCHASTIC_TOURNAMENT): ").strip().upper()
    assert SELECTION_METHOD in {"RWS", "SUS", "TOURNAMENT", "STOCHASTIC_TOURNAMENT"}

except AssertionError:
    print("Invalid input! Please choose a valid option.")
    exit()

# Individual
class Individual:
    def __init__(self, value=None):
        self.genes = value or self._generate_random_genes()
        self.fitness = 0
        self.age = 0

    @staticmethod
    def _generate_random_genes():
        return ''.join(random.choice(string.printable[:95]) for _ in range(len(TARGET_STRING)))

# Population functions
def create_population():
    return [Individual() for _ in range(POPULATION_SIZE)]

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

#Genetic Diversity Metrics
def average_hamming_distance(population):
    total = 0
    count = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            dist = sum(a != b for a, b in zip(population[i].genes, population[j].genes))        #counting different chrom for pairs
            total += dist
            count += 1      #instead of pre calculating the total number of combinations (n choose 2) we count each comparison indivisually 
    return total / count

def count_unique_alleles(population):
    gene_length = len(population[0].genes)
    total_unique = 0
    for pos in range(gene_length):
        alleles = set(ind.genes[pos] for ind in population)     #counting unique chromosomes in on the same index
        total_unique += len(alleles)
    return total_unique

def shannon_entropy(population):
    gene_length = len(population[0].genes)
    total_entropy = 0

    for position in range(gene_length):
        allele_counts = {}  
        for individual in population:       #counting frequency of each char at position i
            allele = individual.genes[position]
            if allele not in allele_counts:
                allele_counts[allele] = 0
            allele_counts[allele] += 1

        # finding the probability for each char
        probabilities = [count / POPULATION_SIZE for count in allele_counts.values()]

        # compute entropy on this position
        entropy_at_position = 0
        for p in probabilities:
            if p > 0:
                entropy_at_position -= p * np.log2(p)

        total_entropy += entropy_at_position

    # average entropy over all positions
    average_entropy = total_entropy / gene_length
    return average_entropy

def roulette_wheel_selection(population, cumulative_scaled, total_scaled): #we pre-calculate the cumulative scaled and total scaled at thr beginig of each generation to prevent redundant calculations
    pick = random.uniform(0, total_scaled)
    for i, cumulative in enumerate(cumulative_scaled): # we iterate through the cumulative scaled until we reach to a cumulative index where pick become larger
        if pick <= cumulative:
            return population[i]

def stochastic_universal_sampling(population, scaled_fitness, total_scaled):
    point_distance = total_scaled / 2  # since we pick 2 parents
    start_point = random.uniform(0, point_distance)
    points = [start_point + i * point_distance for i in range(2)]

    parents = []
    cumulative = 0
    index = 0
    for i, score in enumerate(scaled_fitness):
        cumulative += score
        while index < len(points) and cumulative >= points[index]:
            parents.append(population[i])
            index += 1
    return parents

def tournament_selection(population, k):
    competitors = random.sample(population, k)
    competitors.sort(key=lambda x: x.fitness)
    return competitors[0]

def stochastic_tournament_selection(population, k, p):
    competitors = random.sample(population, k)
    competitors.sort(key=lambda x: x.fitness)
    if random.random() < p:
        return competitors[0]
    else:
        return random.choice(competitors[1:])

def select_parents(population, scaled_fitness=None, cumulative_scaled=None, total_scaled=None):
    if SELECTION_METHOD == "RWS":
        return roulette_wheel_selection(population, cumulative_scaled, total_scaled)
    elif SELECTION_METHOD == "TOURNAMENT":
        return tournament_selection(population, TOURNAMENT_K)
    elif SELECTION_METHOD == "STOCHASTIC_TOURNAMENT":
        return stochastic_tournament_selection(population, TOURNAMENT_K, STOCHASTIC_P)
    else:
        raise ValueError("Invalid selection method or SUS should be handled in crossover()")

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

def crossover(current_population, next_generation, scaled_fitness=None, cumulative_scaled=None, total_scaled=None):
    elite_size = int(POPULATION_SIZE * ELITISM_RATE)
    apply_elitism(current_population, next_generation, elite_size)

    for i in range(elite_size, POPULATION_SIZE):
        if SELECTION_METHOD == "SUS":
        # Select until both parents are young enough
            while True:
                parents = stochastic_universal_sampling(current_population, scaled_fitness, total_scaled)
                parent1, parent2 = parents[0], parents[1]
                if parent1.age <= AGING_MAX_AGE and parent2.age <= AGING_MAX_AGE:
                    break
        else:
            parent1 = select_parents(current_population, scaled_fitness, cumulative_scaled, total_scaled)
            parent2 = select_parents(current_population, scaled_fitness, cumulative_scaled, total_scaled)

            while parent1.age > AGING_MAX_AGE:
                parent1 = select_parents(current_population, scaled_fitness, cumulative_scaled, total_scaled)

            while parent2.age > AGING_MAX_AGE:
                parent2 = select_parents(current_population, scaled_fitness, cumulative_scaled, total_scaled)

        genes1 = parent1.genes
        genes2 = parent2.genes

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
        start_ticks = time.process_time()

        evaluate_population(current_generation)
        sort_population(current_generation)

        #avg_hamming = average_hamming_distance(current_generation)
        unique_alleles = count_unique_alleles(current_generation)
        entropy = shannon_entropy(current_generation)

        for ind in current_generation:
            ind.age += 1

        fitness_values = [ind.fitness for ind in current_generation]
        fitness_log.append(fitness_values.copy())

        if FITNESS_MODE == "lcs" or "combined":
            min_fitness = min(fitness_values)
            fitness_values = [f - min_fitness + 1 for f in fitness_values]
        
        # Precompute scaled fitness and cumulative sums for RWS/SUS (once per generation)
        scaled_fitness = []
        cumulative_scaled = []
        total_scaled = 0

        if SELECTION_METHOD in {"RWS", "SUS"}:
            scaled_fitness = [1 / (f + 1) for f in fitness_values]
            total_scaled = sum(scaled_fitness)
            cumulative_scaled = np.cumsum(scaled_fitness).tolist()

        best = fitness_values[0]
        worst = fitness_values[-1]
        avg = np.mean(fitness_values)
        std_dev = np.std(fitness_values)                  
        variance = np.var(fitness_values)                 
        elapsed_time = time.time() - start_time
        end_ticks = time.process_time()
        clock_ticks = end_ticks - start_ticks

        #Top-Average Selection Probability Ratio
        top_n = int(POPULATION_SIZE * ELITISM_RATE)
        top_avg = np.mean(fitness_values[:top_n])
        top_avg_ratio = top_avg / avg 

        stats_log.append({
            "Generation": gen,
            "Best Fitness": best,
            "Worst Fitness": worst,
            "Average Fitness": avg,
            "Std Dev": std_dev,
            "Fitness Variance": variance,
            "Top-Average Ratio": top_avg_ratio,
            #"Avg Hamming Distance": avg_hamming,
            "Unique Alleles": unique_alleles,
            "Shannon Entropy": entropy,
            "Elapsed Time (s)": elapsed_time,
            "Clock Ticks": clock_ticks
        })

        print(
            f"Generation {gen}: "
            f"Best={best}, Avg={avg:.2f}, Worst={worst}, "
            f"StdDev={std_dev:.2f}, Variance={variance:.2f}, "
            f"Top-Avg Ratio={top_avg_ratio:.3f}, "
            f"UniqueAlleles={unique_alleles}, Entropy={entropy:.3f}, "
            f"Elapsed={elapsed_time:.5f}s"
        )
        print(f"Best match: {current_generation[0].genes}")

        if current_generation[0].genes == TARGET_STRING:
            print("Target string reached! Stopping early.")
            break

        if best_fitness is None or best < best_fitness:
            best_fitness = best
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= NO_IMPROVEMENT_LIMIT:
                print(f"No improvement in {NO_IMPROVEMENT_LIMIT} generations. Stopping early.")
                break

        crossover(current_generation, next_generation, scaled_fitness, cumulative_scaled, total_scaled)
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
