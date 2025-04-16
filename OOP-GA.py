import time
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


# ============================
# Abstract base class for problems
# ============================
class BaseProblem:
    def __init__(self, population_size, max_generations, elitism_ratio, mutation_rate,
                 no_improvement_limit, aging_max_age, tournament_k, stochastic_p):
        self.population_size = population_size
        self.max_generations = max_generations
        self.elitism_ratio = elitism_ratio
        self.mutation_rate = mutation_rate
        self.no_improvement_limit = no_improvement_limit
        self.aging_max_age = aging_max_age
        self.tournament_k = tournament_k
        self.stochastic_p = stochastic_p

    def create_individual(self):
        raise NotImplementedError

    def evaluate_fitness(self, individual):
        raise NotImplementedError

    def get_gene_length(self):
        raise NotImplementedError


class WordMatchProblem(BaseProblem):
    def __init__(self, target, fitness_mode, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.length = len(target)
        self.fitness_mode = fitness_mode

    def create_individual(self):
        return ''.join(random.choice(string.printable[:95]) for _ in range(self.length))

    def evaluate_fitness(self, genes):
        if self.fitness_mode == "ascii":
            return sum(abs(ord(genes[i]) - ord(self.target[i])) for i in range(self.length))
        elif self.fitness_mode == "lcs":
            return -self.lcs_length(genes, self.target)
        elif self.fitness_mode == "combined":
            ascii_distance = sum(abs(ord(genes[i]) - ord(self.target[i])) for i in range(self.length))
            lcs_score = self.lcs_length(genes, self.target)
            return ascii_distance - (lcs_score * 2)

    def get_gene_length(self):
        return self.length

    @staticmethod
    def lcs_length(a, b):
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])
        return table[-1][-1]





class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None
        self.age = 0


class GeneticAlgorithm:
    def __init__(self, problem: BaseProblem, selection_method: str, crossover_type: str):
        self.problem = problem
        self.selection_method = selection_method.upper()
        self.crossover_type = crossover_type.upper()

    def create_population(self):
        return [Individual(self.problem.create_individual()) for _ in range(self.problem.population_size)]

    def evaluate_population(self, population):
        for individual in population:
            individual.fitness = self.problem.evaluate_fitness(individual.genes)

    def mutate(self, individual):
        if isinstance(individual.genes, str):
            index = random.randint(0, len(individual.genes) - 1)
            genes = list(individual.genes)
            genes[index] = random.choice(string.printable[:95])
            individual.genes = ''.join(genes)
        else:
            index = random.randint(0, len(individual.genes) - 1)
            individual.genes[index] = random.randint(0, len(individual.genes) - 1)

    def crossover(self, g1, g2):
        if isinstance(g1, str):
            if self.crossover_type == "SINGLE":
                point = random.randint(1, len(g1) - 1)
                return g1[:point] + g2[point:]
            elif self.crossover_type == "TWO":
                p1 = random.randint(0, len(g1) - 2)
                p2 = random.randint(p1 + 1, len(g1) - 1)
                return g1[:p1] + g2[p1:p2] + g1[p2:]
            elif self.crossover_type == "UNIFORM":
                return ''.join(random.choice([a, b]) for a, b in zip(g1, g2))
        else:
            return [random.choice([a, b]) for a, b in zip(g1, g2)]

    def select_parent(self, population, scaled, cumulative, total):
        if self.selection_method == "RWS":
            pick = random.uniform(0, total)
            for i, cum in enumerate(cumulative):
                if pick <= cum:
                    return population[i]
        elif self.selection_method == "TOURNAMENT":
            return sorted(random.sample(population, self.problem.tournament_k), key=lambda x: x.fitness)[0]
        elif self.selection_method == "STOCHASTIC_TOURNAMENT":
            competitors = sorted(random.sample(population, self.problem.tournament_k), key=lambda x: x.fitness)
            return competitors[0] if random.random() < self.problem.stochastic_p else random.choice(competitors[1:])
        elif self.selection_method == "SUS":
            spacing = total / 2
            start = random.uniform(0, spacing)
            pointers = [start + i * spacing for i in range(2)]
            selected, cumulative_sum, pointer_index = [], 0, 0
            for i, fitness_val in enumerate(scaled):
                cumulative_sum += fitness_val
                while pointer_index < len(pointers) and cumulative_sum >= pointers[pointer_index]:
                    selected.append(population[i])
                    pointer_index += 1
            return selected[0]

    def run(self):
        pop = self.create_population()
        next_gen = [Individual(self.problem.create_individual()) for _ in range(self.problem.population_size)]

        stats, fit_log = [], []
        best_fitness, no_improve = None, 0

        for gen in range(self.problem.max_generations):
            self.evaluate_population(pop)
            pop.sort(key=lambda x: x.fitness)

            fitness_vals = [ind.fitness for ind in pop]
            fit_log.append(fitness_vals.copy())

            if best_fitness is None or fitness_vals[0] < best_fitness:
                best_fitness = fitness_vals[0]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.problem.no_improvement_limit:
                    print("Stopping early due to no improvement.")
                    break

            if isinstance(pop[0].genes, str):
                print(f"Gen {gen}: best={fitness_vals[0]} match={pop[0].genes}")
            else:
                print(f"Gen {gen}: best={fitness_vals[0]}")

            if self.selection_method in {"RWS", "SUS"}:
                scaled = [1 / (f + 1) for f in fitness_vals]
                total = sum(scaled)
                cumulative = np.cumsum(scaled).tolist()
            else:
                scaled, cumulative, total = [], [], 0

            elite_size = int(self.problem.population_size * self.problem.elitism_ratio)
            for i in range(elite_size):
                next_gen[i].genes = pop[i].genes
                next_gen[i].fitness = pop[i].fitness

            for i in range(elite_size, self.problem.population_size):
                p1 = self.select_parent(pop, scaled, cumulative, total)
                p2 = self.select_parent(pop, scaled, cumulative, total)

                while p1.age > self.problem.aging_max_age:
                    p1 = self.select_parent(pop, scaled, cumulative, total)
                while p2.age > self.problem.aging_max_age:
                    p2 = self.select_parent(pop, scaled, cumulative, total)

                child_genes = self.crossover(p1.genes, p2.genes)
                child = Individual(child_genes)
                if random.random() < self.problem.mutation_rate:
                    self.mutate(child)
                next_gen[i] = child

            for ind in pop:
                ind.age += 1

            pop, next_gen = next_gen, pop
        if isinstance(pop[0].genes, str):
            print(f"Gen {gen}: best={fitness_vals[0]} match={pop[0].genes}")
        else:
            print(f"Gen {gen}: best={fitness_vals[0]} bins={pop[0].genes}")

        return pd.DataFrame(stats), fit_log



def best_fit_bin_packing(items, bin_capacity):
    """
    Places each item in the bin with the least remaining space that can still accommodate it.
    Returns the number of bins used.
    """
    bins = []

    for item in items:
        # Find the best bin that can accommodate the item
        min_space = bin_capacity + 1
        best_bin_index = -1

        for i, b in enumerate(bins):
            remaining = bin_capacity - sum(b)
            if item <= remaining and remaining < min_space:
                min_space = remaining
                best_bin_index = i

        if best_bin_index == -1:
            # No bin can accommodate the item, open a new one
            bins.append([item])
        else:
            bins[best_bin_index].append(item)

    return len(bins)

# =========================
# Choose and initialize problem
# =========================
problem_choice = input("Choose problem (word / binpack): ").strip().lower()

if problem_choice == "word":
    target = "loay,mohammad"
    fitness_mode = input("Choose fitness mode (ascii / lcs / combined): ").strip().lower()
    problem = WordMatchProblem(target=target, fitness_mode=fitness_mode,
        population_size=512, max_generations=500, elitism_ratio=0.1, mutation_rate=0.25,
        no_improvement_limit=50, aging_max_age=10, tournament_k=5, stochastic_p=0.8)

elif problem_choice == "binpack":
    u120_00_capacity = 150
    u120_00 = [
        42, 69, 67, 57, 93, 90, 38, 36, 45, 42, 33, 79, 27, 57, 44, 84, 86, 92, 46, 38,
        85, 33, 82, 73, 49, 70, 59, 23, 57, 72, 74, 69, 33, 42, 28, 46, 30, 64, 29, 74,
        41, 49, 55, 98, 80, 32, 25, 38, 82, 30, 35, 39, 57, 84, 62, 50, 55, 27, 30, 36,
        20, 78, 47, 26, 45, 41, 58, 98, 91, 96, 73, 84, 37, 93, 91, 43, 73, 85, 81, 79,
        71, 80, 76, 83, 41, 78, 70, 23, 42, 87, 43, 84, 60, 55, 49, 78, 73, 62, 36, 44,
        94, 69, 32, 96, 70, 84, 58, 78, 25, 80, 58, 66, 83, 24, 98, 60, 42, 43, 43, 39
    ]

    u120_01_capacity = 150
    u120_01 = [
        97, 57, 81, 62, 75, 81, 23, 43, 50, 38, 60, 58, 70, 88, 36, 90, 37, 45, 45, 39,
        44, 53, 70, 24, 82, 81, 47, 97, 35, 65, 74, 68, 49, 55, 52, 94, 95, 29, 99, 20,
        22, 25, 49, 46, 98, 59, 98, 60, 23, 72, 33, 98, 80, 95, 78, 57, 67, 53, 47, 53,
        36, 38, 92, 30, 80, 32, 97, 39, 80, 72, 55, 41, 60, 67, 53, 65, 95, 20, 66, 78,
        98, 47, 100, 85, 53, 53, 67, 27, 22, 61, 43, 52, 76, 64, 61, 29, 30, 46, 79, 66,
        27, 79, 98, 90, 22, 75, 57, 67, 36, 70, 99, 48, 43, 45, 71, 100, 88, 48, 27, 39
    ]

    u120_02_capacity = 150
    u120_02 = [
        38, 100, 60, 42, 20, 69, 24, 23, 92, 32, 84, 36, 65, 84, 34, 68, 64, 33, 69, 27,
        47, 21, 85, 88, 59, 61, 50, 53, 37, 75, 64, 84, 74, 57, 83, 28, 31, 97, 61, 36,
        46, 37, 96, 80, 53, 51, 68, 90, 64, 81, 66, 67, 80, 37, 92, 67, 64, 31, 94, 45,
        80, 28, 76, 29, 64, 38, 48, 40, 29, 44, 81, 35, 51, 48, 67, 24, 46, 38, 76, 22,
        30, 67, 45, 41, 29, 41, 79, 21, 25, 90, 62, 34, 73, 50, 79, 66, 59, 42, 90, 79,
        70, 66, 80, 35, 62, 98, 97, 37, 32, 75, 91, 91, 48, 26, 23, 32, 100, 46, 29, 26
    ]

    u120_03_capacity = 150
    u120_03 = [
        29, 26, 83, 82, 92, 95, 87, 63, 57, 100, 63, 65, 81, 46, 42, 95, 90, 80, 53, 27,
        84, 40, 22, 97, 20, 73, 63, 95, 46, 42, 47, 40, 26, 88, 49, 24, 92, 87, 68, 95,
        34, 82, 84, 43, 54, 73, 66, 32, 62, 48, 99, 90, 86, 28, 25, 25, 89, 67, 96, 35,
        33, 70, 40, 59, 32, 94, 34, 86, 35, 45, 25, 76, 80, 42, 91, 44, 91, 97, 60, 29,
        45, 37, 61, 54, 78, 56, 74, 74, 45, 21, 96, 37, 75, 100, 58, 84, 85, 56, 54, 71,
        52, 79, 43, 35, 27, 70, 31, 47, 35, 26, 30, 97, 90, 80, 58, 60, 73, 46, 71, 39
    ]

    u120_04_capacity = 150
    u120_04 = [
        42, 98, 27, 21, 71, 71, 78, 76, 57, 24, 91, 84, 35, 25, 77, 96, 97, 89, 30, 86,
        81, 39, 75, 66, 85, 36, 60, 56, 50, 75, 75, 37, 87, 95, 21, 99, 42, 57, 31, 37,
        42, 40, 69, 91, 45, 97, 84, 90, 52, 43, 68, 53, 37, 65, 79, 73, 92, 87, 20, 20,
        73, 42, 52, 20, 24, 76, 71, 72, 21, 21, 82, 92, 78, 87, 50, 41, 31, 73, 89, 59,
        88, 40, 71, 69, 45, 57, 49, 68, 84, 32, 69, 77, 92, 98, 57, 39, 32, 23, 99, 91,
        48, 21, 70, 43, 73, 69, 65, 57, 67, 28, 84, 42, 61, 92, 82, 34, 74, 55, 60, 69
    ]

    # Define the list of problems
binpack_problems = [
    ("u120_00", u120_00, u120_00_capacity),
    ("u120_01", u120_01, u120_01_capacity),
    ("u120_02", u120_02, u120_02_capacity),
    ("u120_03", u120_03, u120_03_capacity),
    ("u120_04", u120_04, u120_04_capacity)
]

# Run both GA and Best-Fit, collect results
results = []

for name, items, capacity in binpack_problems:
    print(f"\nRunning Problem: {name}")
    
    # Genetic Algorithm
    problem = BinPackingProblem(
        items=items,
        bin_capacity=capacity,
        population_size=500,
        max_generations=1000,
        elitism_ratio=0.1,
        mutation_rate=0.25,
        no_improvement_limit=500,
        aging_max_age=8,
        tournament_k=5,
        stochastic_p=0.75
    )
    ga = GeneticAlgorithm(problem, selection_method="TOURNAMENT", crossover_type="UNIFORM")
    _, fit_log = ga.run()
    ga_result = min(fit_log[-1]) if fit_log else None

    # Best-Fit Heuristic
    bf_result = best_fit_bin_packing(items, capacity)

    results.append({
        "Problem": name,
        "GA Bins Used": ga_result,
        "Best-Fit Bins Used": bf_result
    })

    # Display nicely
    df = pd.DataFrame(results)
    print("\n===== Bin Packing Results Comparison =====")
    print(df.to_string(index=False))

else:
    raise ValueError("Invalid problem type")

crossover_type = input("Enter crossover type (SINGLE / TWO / UNIFORM): ").strip().upper()
selection_method = input("Enter selection method (RWS / SUS / TOURNAMENT / STOCHASTIC_TOURNAMENT): ").strip().upper()

# Run the algorithm
ga = GeneticAlgorithm(problem, selection_method, crossover_type)
results_df, fitness_log = ga.run()

# Optionally: Plot fitness log for visualization (only useful for word match or numeric fitness)
if fitness_log and isinstance(fitness_log[0], list):
    plt.figure(figsize=(12, 6))
    plt.boxplot(fitness_log, vert=True, patch_artist=True, showfliers=True)
    plt.title('Fitness Distribution Per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.tight_layout()
    plt.show()