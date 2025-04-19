import sys
import time
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# =============================================================================
# GA Input Loader
# =============================================================================
def load_ga_input(path: str):
    """
    Reads a single TXT file in this exact format:
      Line 1: exactly “###”                         <-- file‐format check
      Line 2: target string for the word problem
      Line 3: three ints: <bin_capacity> <num_items> <theoretical_best_bins>
      Lines 4+: one item weight (int) per line
    Returns:
      target (str),
      bin_capacity (int),
      items (list[int]),
      theoretical_best (int)
    """
    with open(path, 'r', encoding='utf-8') as f:
        # drop any empty lines but preserve order
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]

    # 1) sanity check header
    if not lines or lines[0].strip() != "###":
        raise ValueError("Input file must start with ### on line 1")

    # 2) ensure at least 3 lines
    if len(lines) < 3:
        raise ValueError("Input file too short; needs at least 3 lines")

    # 3) line 2 is the word target
    target = lines[1]

    # 4) line 3 has capacity, count, theoretical best
    parts = lines[2].split()
    if len(parts) != 3:
        raise ValueError("Line 3 must have exactly 3 numbers: capacity, num_items, theoretical_best")
    bin_capacity, num_items, theoretical_best = map(int, parts)

    # 5) remaining lines are item weights
    items = [int(x) for x in lines[3:]]
    if len(items) != num_items:
        print(f"Warning: declared {num_items} items but found {len(items)} lines")

    return target, bin_capacity, items, theoretical_best


# =========================
# Abstract base class
# =========================
class BaseProblem:
    def __init__(self, population_size, max_generations, elitism_ratio, mutation_rate,
                 no_improvement_limit, aging_max_age, tournament_k, stochastic_p):
        self.population_size      = population_size
        self.max_generations      = max_generations
        self.elitism_ratio        = elitism_ratio
        self.mutation_rate        = mutation_rate
        self.no_improvement_limit = no_improvement_limit
        self.aging_max_age        = aging_max_age
        self.tournament_k         = tournament_k
        self.stochastic_p         = stochastic_p

    def create_individual(self):
        raise NotImplementedError

    def evaluate_fitness(self, individual):
        raise NotImplementedError

    def get_gene_length(self):
        raise NotImplementedError


# =========================
# Word‐matching problem
# =========================
class WordMatchProblem(BaseProblem):
    def __init__(self, target, fitness_mode, **kwargs):
        super().__init__(**kwargs)
        self.target       = target
        self.length       = len(target)
        self.fitness_mode = fitness_mode

    def create_individual(self):
        return ''.join(random.choice(string.printable[:95]) for _ in range(self.length))

    def evaluate_fitness(self, genes):
        if self.fitness_mode == "ascii":
            return sum(abs(ord(genes[i]) - ord(self.target[i])) for i in range(self.length))
        elif self.fitness_mode == "lcs":
            return -self.lcs_length(genes, self.target)
        else:  # combined
            ascii_dist = sum(abs(ord(genes[i]) - ord(self.target[i])) for i in range(self.length))
            lcs_score  = self.lcs_length(genes, self.target)
            return ascii_dist - (2 * lcs_score)

    def get_gene_length(self):
        return self.length

    @staticmethod
    def lcs_length(a, b):
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                if a[i-1] == b[j-1]:
                    table[i][j] = table[i-1][j-1] + 1
                else:
                    table[i][j] = max(table[i-1][j], table[i][j-1])
        return table[-1][-1]


# =========================
# Bin‐packing problem
# =========================
class BinPackingProblem(BaseProblem):
    def __init__(self, items, bin_capacity, **kwargs):
        super().__init__(**kwargs)
        self.items        = items
        self.bin_capacity = bin_capacity
        self.num_items    = len(items)

    def create_individual(self):
        return random.sample(range(self.num_items), self.num_items)

    def evaluate_fitness(self, individual):
        bins, loads = [], []
        for idx in individual:
            size = self.items[idx]
            placed = False
            for i in range(len(bins)):
                if loads[i] + size <= self.bin_capacity:
                    bins[i].append(idx)
                    loads[i] += size
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                loads.append(size)
        return len(bins)

    def get_gene_length(self):
        return self.num_items


# =========================
# Individual wrapper
# =========================
class Individual:
    def __init__(self, genes):
        self.genes   = genes
        self.fitness = None
        self.age     = 0


# =========================
# Core GA engine
# =========================
class GeneticAlgorithm:
    def __init__(self, problem: BaseProblem, selection_method: str, crossover_type: str):
        self.problem          = problem
        self.selection_method = selection_method.upper()
        self.crossover_type   = crossover_type.upper()

    def create_population(self):
        return [Individual(self.problem.create_individual())
                for _ in range(self.problem.population_size)]

    def evaluate_population(self, pop):
        for ind in pop:
            ind.fitness = self.problem.evaluate_fitness(ind.genes)

    def mutate(self, ind: Individual):
        genes = ind.genes
        if isinstance(genes, str):
            i = random.randrange(len(genes))
            lst = list(genes)
            lst[i] = random.choice(string.printable[:95])
            ind.genes = ''.join(lst)
        else:
            a, b = random.sample(range(len(genes)), 2)
            genes[a], genes[b] = genes[b], genes[a]

    def crossover(self, g1, g2):
        if isinstance(g1, str):
            if self.crossover_type == "SINGLE":
                p = random.randint(1, len(g1)-1)
                return g1[:p] + g2[p:]
            elif self.crossover_type == "TWO":
                p1 = random.randint(0, len(g1)-2)
                p2 = random.randint(p1+1, len(g1)-1)
                return g1[:p1] + g2[p1:p2] + g1[p2:]
            else:  # UNIFORM
                return ''.join(random.choice([a,b]) for a,b in zip(g1,g2))
        else:
            size  = len(g1)
            child = [None]*size
            s,e   = sorted(random.sample(range(size),2))
            child[s:e+1] = g1[s:e+1]
            ptr = 0
            for gene in g2:
                if gene not in child:
                    while child[ptr] is not None:
                        ptr += 1
                    child[ptr] = gene
            return child

    def select_parent(self, pop, scaled, cumulative, total):
        if self.selection_method == "RWS":
            pick = random.uniform(0, total)
            for i, cum in enumerate(cumulative):
                if pick <= cum:
                    return pop[i]
        elif self.selection_method == "TOURNAMENT":
            return min(random.sample(pop, self.problem.tournament_k),
                       key=lambda x: x.fitness)
        elif self.selection_method == "STOCHASTIC_TOURNAMENT":
            group = sorted(random.sample(pop, self.problem.tournament_k),
                           key=lambda x: x.fitness)
            return group[0] if random.random() < self.problem.stochastic_p \
                             else random.choice(group[1:])
        else:  # SUS
            spacing = total / 2
            start   = random.uniform(0, spacing)
            pointers = [start + i*spacing for i in range(2)]
            selected, cum_sum, idx = [], 0, 0
            for i,val in enumerate(scaled):
                cum_sum += val
                while idx < len(pointers) and cum_sum >= pointers[idx]:
                    selected.append(pop[i])
                    idx += 1
            return selected[0]

    def run(self):
        pop      = self.create_population()
        next_gen = [Individual(self.problem.create_individual())
                    for _ in range(self.problem.population_size)]
        fit_log, best_fit, no_imp = [], None, 0

        for gen in range(self.problem.max_generations):
            self.evaluate_population(pop)
            pop.sort(key=lambda x: x.fitness)

            # target‐match stop for word‐problem
            if isinstance(pop[0].genes, str) and pop[0].genes == self.problem.target:
                print(f"Matched '{pop[0].genes}' at generation {gen}")
                break

            vals = [ind.fitness for ind in pop]
            fit_log.append(vals.copy())

            # early‐stop on no improvement
            if best_fit is None or vals[0] < best_fit:
                best_fit, no_imp = vals[0], 0
            else:
                no_imp += 1
                if no_imp >= self.problem.no_improvement_limit:
                    print("No improvement—stopping early.")
                    break

            # print progress
            if isinstance(pop[0].genes, str):
                print(f"Gen {gen}: best={vals[0]} match={pop[0].genes}")
            else:
                print(f"Gen {gen}: best bins={vals[0]}")

            # prepare selection weights if needed
            if self.selection_method in ("RWS","SUS"):
                min_val = min(vals)
                offset  = -min_val+1 if min_val <= 0 else 0
                scaled  = [1/(v+offset) for v in vals]
                total   = sum(scaled)
                cumulative = np.cumsum(scaled).tolist()
            else:
                scaled, cumulative, total = [], [], 0

            # copy elites
            elite_size = int(self.problem.population_size * self.problem.elitism_ratio)
            for i in range(elite_size):
                next_gen[i].genes   = pop[i].genes
                next_gen[i].fitness = pop[i].fitness

            # fill rest by crossover+mutation
            for i in range(elite_size, self.problem.population_size):
                p1 = self.select_parent(pop, scaled, cumulative, total)
                p2 = self.select_parent(pop, scaled, cumulative, total)
                while p1.age > self.problem.aging_max_age:
                    p1 = self.select_parent(pop, scaled, cumulative, total)
                while p2.age > self.problem.aging_max_age:
                    p2 = self.select_parent(pop, scaled, cumulative, total)

                child = Individual(self.crossover(p1.genes, p2.genes))
                if random.random() < self.problem.mutation_rate:
                    self.mutate(child)
                next_gen[i] = child

            # age bump
            for ind in pop:
                ind.age += 1

            pop, next_gen = next_gen, pop

        # final report
        if isinstance(pop[0].genes, str):
            print(f"Final: best={pop[0].fitness}, match={pop[0].genes}")
        else:
            print(f"Final: best bins={pop[0].fitness}")

        return pd.DataFrame(), fit_log


# =============================================================================
# Main entry: “word” or “binpack”
# =============================================================================
if __name__ == '__main__':
    raw = input("Path to your GA‑input file: ").strip()
    input_path = raw.strip('"').strip("'")

    try:
        target, bin_capacity, items, theoretical_best = load_ga_input(input_path)
    except Exception as e:
        print("Error loading file:", e)
        sys.exit(1)

    choice = input("Which problem to run? (word / binpack): ").strip().lower()
    if choice == 'word':
        mode   = input("Fitness mode (ascii / lcs / combined): ").strip().lower()
        xover  = input("Crossover (SINGLE / TWO / UNIFORM): ").strip().upper()
        select = input("Selection (RWS / SUS / TOURNAMENT / STOCHASTIC_TOURNAMENT): ").strip().upper()

        prob = WordMatchProblem(
            target=target, fitness_mode=mode,
            population_size=512, max_generations=500,
            elitism_ratio=0.1, mutation_rate=0.25,
            no_improvement_limit=50, aging_max_age=10,
            tournament_k=5, stochastic_p=0.8
        )
        ga = GeneticAlgorithm(prob, select, xover)
        ga.run()

    elif choice == 'binpack':
        xover  = input("Crossover (SINGLE / TWO / UNIFORM): ").strip().upper()
        select = input("Selection (RWS / SUS / TOURNAMENT / STOCHASTIC_TOURNAMENT): ").strip().upper()

        prob = BinPackingProblem(
            items=items, bin_capacity=bin_capacity,
            population_size=500, max_generations=1000,
            elitism_ratio=0.1, mutation_rate=0.25,
            no_improvement_limit=200, aging_max_age=8,
            tournament_k=5, stochastic_p=0.75
        )
        ga = GeneticAlgorithm(prob, select, xover)
        ga.run()
        print(f"\nTheoretical best‑possible bins for this instance: {theoretical_best}")


    else:
        print("Invalid choice; must be 'word' or 'binpack'")
        sys.exit(1)
