import random
import numpy as np

# Problem definition
CAPACITY = 150
weights =  [
    145, 100, 60, 42, 20, 69, 24, 23, 92, 32, 84, 36, 65, 84, 34, 68, 64, 33, 69, 27,
    47, 21, 85, 88, 59, 61, 50, 53, 37, 75, 64, 84, 74, 57, 83, 28, 31, 97, 61, 36,
    46, 37, 96, 80, 53, 51, 68, 90, 64, 81, 66, 67, 80, 37, 92, 67, 64, 31, 94, 45,
    80, 28, 76, 29, 64, 38, 48, 40, 29, 44, 81, 35, 51, 48, 67, 24, 46, 38, 76, 22,
    30, 67, 45, 41, 29, 41, 79, 21, 25, 90, 62, 34, 73, 50, 79, 66, 59, 42, 90, 79,
    70, 66, 80, 35, 62, 98, 97, 37, 32, 75, 91, 91, 48, 26, 23, 32, 100, 46, 29, 26
]
  # replace with actual input
NUM_ITEMS = len(weights)
OPTIMAL_BINS = 46  # Replace with known optimal if needed

# Best Fit Algorithm
bins = []

for weight in weights:
    min_space = float('inf')
    chosen_bin = -1
    for i in range(len(bins)):
        if bins[i] + weight <= CAPACITY and CAPACITY - (bins[i] + weight) < min_space:
            min_space = CAPACITY - (bins[i] + weight)
            chosen_bin = i
    if chosen_bin == -1:
        bins.append(weight)  # Open new bin
    else:
        bins[chosen_bin] += weight  # Add to best fitting bin

print(f"Total bins used (Best Fit): {len(bins)}")

# GA parameters
POP_SIZE = 200
MAX_GENERATIONS = 1000
MUTATION_RATE = 0.5
ELITE_COUNT = 20
TOURNAMENT_K = 10
CROSSOVER_TYPE = "uniform"  # "uniform", "single", "two"
SELECTION_METHOD = "rws"  # "tournament", "rws"

# --- Core functions --- #

def evaluate(chrom):
    bins = {}
    for item, bin_id in enumerate(chrom):
        bins.setdefault(bin_id, []).append(item)
    bin_weights = {b: sum(weights[i] for i in items) for b, items in bins.items()}
    overfilled = sum(1 for w in bin_weights.values() if w > CAPACITY)
    return len(bin_weights) + (1000 * overfilled)

def initialize_population(size):
    population = []

    # --- Add Best-Fit First individual ---
    bff_bins = []
    bff_chrom = [None] * NUM_ITEMS
    for item_idx, w in enumerate(weights):
        best_bin = -1
        min_space = float('inf')
        for b_idx in range(len(bff_bins)):
            space_left = CAPACITY - bff_bins[b_idx]
            if w <= space_left and space_left < min_space:
                best_bin = b_idx
                min_space = space_left
        if best_bin == -1:
            bff_bins.append(w)
            bff_chrom[item_idx] = len(bff_bins) - 1
        else:
            bff_bins[best_bin] += w
            bff_chrom[item_idx] = best_bin

    population.append(bff_chrom)

    # --- Fill the rest of the population with random + repaired ---
    for _ in range(size - 1):
        chrom = [random.randint(0, NUM_ITEMS - 1) for _ in range(NUM_ITEMS)]
        chrom = repair(chrom)
        population.append(chrom)

    return population


def repair(chrom):
    bins = {}
    for item, b in enumerate(chrom):
        bins.setdefault(b, []).append(item)
    new_chrom = [None] * NUM_ITEMS
    bin_counter = 0
    for items in bins.values():
        current_bin_weight = 0
        for item in items:
            if current_bin_weight + weights[item] > CAPACITY:
                bin_counter += 1
                current_bin_weight = 0
            current_bin_weight += weights[item]
            new_chrom[item] = bin_counter
        bin_counter += 1
    return new_chrom

def tournament_selection(pop, fits, k=TOURNAMENT_K):
    return min(random.sample(pop, k), key=lambda c: evaluate(c))

def rws_selection(pop, fits):
    max_f = max(fits)
    probs = [(max_f - f + 1) for f in fits]
    total = sum(probs)
    pick = random.uniform(0, total)
    current = 0
    for i, p in enumerate(probs):
        current += p
        if current > pick:
            return pop[i]
    return pop[-1]

def crossover(p1, p2):
    if CROSSOVER_TYPE == "single":
        pt = random.randint(1, NUM_ITEMS - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    elif CROSSOVER_TYPE == "two":
        a, b = sorted(random.sample(range(NUM_ITEMS), 2))
        return (p1[:a] + p2[a:b] + p1[b:], p2[:a] + p1[a:b] + p2[b:])
    elif CROSSOVER_TYPE == "uniform":
        return ([random.choice([i, j]) for i, j in zip(p1, p2)],
                [random.choice([i, j]) for i, j in zip(p2, p1)])
    raise ValueError("Unknown crossover type")

def mutate(chrom):
    if random.random() < MUTATION_RATE:
        for _ in range(random.randint(1, 3)):
            chrom[random.randint(0, NUM_ITEMS - 1)] = random.randint(0, NUM_ITEMS - 1)
    return chrom

# --- GA loop --- #

population = initialize_population(POP_SIZE)
fitnesses = [evaluate(ind) for ind in population]

for generation in range(MAX_GENERATIONS):
    best = min(fitnesses)
    avg = sum(fitnesses) / len(fitnesses)
    print(f"Generation {generation}: Best = {best}, Avg = {avg:.2f}")

    if best == OPTIMAL_BINS:
        print(f"Found optimal solution at generation {generation}")
        break

    sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
    new_pop = [population[i][:] for i in sorted_idx[:ELITE_COUNT]]

    while len(new_pop) < POP_SIZE:
        if SELECTION_METHOD == "tournament":
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
        else:
            p1 = rws_selection(population, fitnesses)
            p2 = rws_selection(population, fitnesses)

        c1, c2 = crossover(p1, p2)
        new_pop.append(repair(mutate(c1)))
        if len(new_pop) < POP_SIZE:
            new_pop.append(repair(mutate(c2)))

    population = new_pop
    fitnesses = [evaluate(ind) for ind in population]
