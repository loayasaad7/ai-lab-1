from copy import deepcopy
import random
from collections import defaultdict

class GroupedBinPackingProblem:
    def __init__(self, items, bin_capacity):
        self.items = items
        self.bin_capacity = bin_capacity
        self.num_items = len(items)

    def create_individual(self):
        # FFD + random shuffle hybrid
        sorted_items = sorted(enumerate(self.items), key=lambda x: -x[1])
        bins = []

        for idx, _ in sorted_items:
            placed = False
            random.shuffle(bins)
            for b in bins:
                if sum(self.items[i] for i in b) + self.items[idx] <= self.bin_capacity:
                    b.append(idx)
                    placed = True
                    break
            if not placed:
                bins.append([idx])
        return bins

    def evaluate_fitness(self, individual):
        return len(individual)

    def is_valid(self, bins):
        for b in bins:
            if sum(self.items[i] for i in b) > self.bin_capacity:
                return False
        return True




class GroupedGeneticAlgorithm:
    def __init__(self, problem, population_size=100, max_generations=1000,
                 mutation_rate=0.2, tournament_k=3):
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k

    def create_population(self):
        return [self.problem.create_individual() for _ in range(self.population_size)]

    def evaluate_population(self, population):
        return [(p, self.problem.evaluate_fitness(p)) for p in population]

    def tournament_select(self, pop_fitness):
        selected = random.sample(pop_fitness, self.tournament_k)
        selected.sort(key=lambda x: x[1])
        return deepcopy(selected[0][0])

    def crossover(self, p1, p2):
        child_bins = []
        used = set()

        for b in p1:
            if random.random() < 0.5:
                valid_bin = [i for i in b if i not in used]
                if valid_bin:
                    child_bins.append(valid_bin)
                    used.update(valid_bin)

        for b in p2:
            valid_bin = [i for i in b if i not in used]
            if valid_bin:
                total = sum(self.problem.items[i] for i in valid_bin)
                if total <= self.problem.bin_capacity:
                    child_bins.append(valid_bin)
                    used.update(valid_bin)

        remaining = [i for i in range(self.problem.num_items) if i not in used]
        random.shuffle(remaining)
        for idx in remaining:
            placed = False
            for b in child_bins:
                if sum(self.problem.items[i] for i in b) + self.problem.items[idx] <= self.problem.bin_capacity:
                    b.append(idx)
                    placed = True
                    break
            if not placed:
                child_bins.append([idx])
        return child_bins

    def mutate(self, individual):
        if len(individual) < 2:
            return
        source = random.choice(individual)
        if not source:
            return
        item = random.choice(source)
        source.remove(item)

        for target in individual:
            if sum(self.problem.items[i] for i in target) + self.problem.items[item] <= self.problem.bin_capacity:
                target.append(item)
                break
        else:
            individual.append([item])
        individual[:] = [b for b in individual if b]

    def run(self):
        pop = self.create_population()
        best_fit = float('inf')
        best_ind = None

        for gen in range(self.max_generations):
            pop_fitness = self.evaluate_population(pop)
            pop_fitness.sort(key=lambda x: x[1])

            if pop_fitness[0][1] < best_fit:
                best_fit = pop_fitness[0][1]
                best_ind = deepcopy(pop_fitness[0][0])
                print(f"Gen {gen}: Best bins = {best_fit}")
            else:
                print(f"Gen {gen}: Still stuck at = {best_fit}")

            next_pop = [deepcopy(pop_fitness[0][0])]  # always keep best from current gen

            while len(next_pop) < self.population_size:
                p1 = self.tournament_select(pop_fitness)
                p2 = self.tournament_select(pop_fitness)
                child = self.crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    self.mutate(child)
                next_pop.append(child)

            pop = next_pop

        return best_ind, best_fit
    



# === Your item list ===
items = [
    38, 100, 60, 42, 20, 69, 24, 23, 92, 32, 84, 36, 65, 84, 34, 68, 64, 33, 69, 27,
    47, 21, 85, 88, 59, 61, 50, 53, 37, 75, 64, 84, 74, 57, 83, 28, 31, 97, 61, 36,
    46, 37, 96, 80, 53, 51, 68, 90, 64, 81, 66, 67, 80, 37, 92, 67, 64, 31, 94, 45,
    80, 28, 76, 29, 64, 38, 48, 40, 29, 44, 81, 35, 51, 48, 67, 24, 46, 38, 76, 22,
    30, 67, 45, 41, 29, 41, 79, 21, 25, 90, 62, 34, 73, 50, 79, 66, 59, 42, 90, 79,
    70, 66, 80, 35, 62, 98, 97, 37, 32, 75, 91, 91, 48, 26, 23, 32, 100, 46, 29, 26
]

capacity = 150
problem = GroupedBinPackingProblem(items, capacity)
ga = GroupedGeneticAlgorithm(problem, population_size=300, max_generations=500, mutation_rate=0.2)

# === Run GA ===
best_solution, best_bins = ga.run()

# === Post-merge optimizer ===
def post_merge(problem, bins):
    merged = True
    while merged:
        merged = False
        for i in range(len(bins)):
            for j in range(i + 1, len(bins)):
                combined = bins[i] + bins[j]
                new_bin = []
                for item in combined:
                    if sum(problem.items[i] for i in new_bin) + problem.items[item] <= problem.bin_capacity:
                        new_bin.append(item)

                if len(new_bin) == len(combined):
                    new_bins = bins[:i] + bins[i+1:j] + bins[j+1:]
                    new_bins.append(new_bin)
                    bins = new_bins
                    merged = True
                    break
            if merged:
                break
    return bins

# === Final rebalancer ===
def rebalance_bins(problem, bins):
    changed = True
    while changed:
        changed = False
        for i in range(len(bins)):
            for j in range(i + 1, len(bins)):
                b1, b2 = bins[i], bins[j]
                combined = b1 + b2
                new_b1, new_b2 = [], []

                for idx in combined:
                    if sum(problem.items[i] for i in new_b1) + problem.items[idx] <= problem.bin_capacity:
                        new_b1.append(idx)
                    elif sum(problem.items[i] for i in new_b2) + problem.items[idx] <= problem.bin_capacity:
                        new_b2.append(idx)
                    else:
                        break
                else:
                    new_bins = bins[:i] + bins[i+1:j] + bins[j+1:]
                    new_bins.append(new_b1)
                    if new_b2:
                        new_bins.append(new_b2)
                    bins = new_bins
                    changed = True
                    break
            if changed:
                break
    return bins

# === Apply both optimizers ===
merged = post_merge(problem, best_solution)
final_bins = rebalance_bins(problem, merged)

# === Final result ===
print(f"\n🧠 Final optimized bin count: {len(final_bins)}")


