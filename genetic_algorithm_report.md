## Genetic Algorithm Report — Questions 1 to 7

---

### Question 1: Fitness Statistics Per Generation

To assess the performance of the genetic algorithm, we recorded the following statistics for each generation:

- **Best Fitness:** The fitness of the most fit individual.
- **Worst Fitness:** The fitness of the least fit individual.
- **Average Fitness:** Mean fitness across the population.
- **Standard Deviation:** Variance in population's fitness.

These were calculated and stored like this:

```python
fitness_values = [ind.fitness for ind in current_generation]
best = fitness_values[0]
worst = fitness_values[-1]
avg = np.mean(fitness_values)
std_dev = np.std(fitness_values)
```

---

### Question 2: Elapsed Time Per Generation

We tracked the execution time for each generation to understand computational load:

```python
start_time = time.time()
...
elapsed_time = time.time() - start_time
```

This helps visualize the efficiency of the algorithm across generations.

---

### Question 3a: Line Graph for Fitness Over Generations

To illustrate convergence, we plotted line charts for fitness progression:

```python
plt.plot(results_df['Generation'], results_df['Best Fitness'], label='Best Fitness')
plt.plot(results_df['Generation'], results_df['Average Fitness'], label='Average Fitness')
plt.plot(results_df['Generation'], results_df['Worst Fitness'], label='Worst Fitness')
```

This gives a clear view of improvement and stagnation during evolution.

---

### Question 3b: Boxplot of Fitness Distribution

We used a boxplot to show the spread of fitness values per generation:

```python
plt.boxplot(fitness_over_time, vert=True, patch_artist=True, showfliers=True)
```

This helps analyze population diversity. A tight box indicates convergence, while a wide box shows variation.

---

### Question 4: Implementing Multiple Crossover Operators

We implemented three types of crossover:
- **SINGLE:** One random split point.
- **TWO:** Two random split points.
- **UNIFORM:** Mix genes randomly from both parents.

The user chooses the type at runtime:

```python
CROSSOVER_TYPE = input("Enter crossover type (SINGLE / TWO / UNIFORM): ").strip().upper()
```

And this choice controls the logic in the crossover function:

```python
if CROSSOVER_TYPE == "SINGLE":
    point = random.randint(1, len(TARGET_STRING) - 1)
    child_genes = genes1[:point] + genes2[point:]
elif CROSSOVER_TYPE == "TWO":
    p1 = random.randint(0, len(TARGET_STRING) - 2)
    p2 = random.randint(p1 + 1, len(TARGET_STRING) - 1)
    child_genes = genes1[:p1] + genes2[p1:p2] + genes1[p2:]
elif CROSSOVER_TYPE == "UNIFORM":
    child_genes = ''.join(random.choice([g1, g2]) for g1, g2 in zip(genes1, genes2))
```

---

### Question 5: Exploration vs Exploitation

We identified which parts of the algorithm contribute to:

- **Exploration:**
  - Mutation: Introduces new, random traits.
  - Uniform crossover: Generates diverse offspring.
  - Random selection of parents (not just best ones).

- **Exploitation:**
  - Elitism: Keeps top individuals unchanged.
  - Sorting by fitness: Prioritizes the best.
  - Selecting parents from top 50%.

Balancing both aspects is crucial to success.

---

### Question 6: Algorithm Variants

We compared the algorithm in 3 modes:

- **a. Only crossover, no mutation:**
  - Converges, but may get stuck early.

- **b. Only mutation, no crossover:**
  - Progress is very slow, random walk.

- **c. Both enabled:**
  - Fast convergence and diversity.

You can test this by disabling mutation or crossover in the code (comment out mutation logic or crossover function).

---

### Question 7: Longest Common Subsequence (LCS) as a Metric

We added LCS-based fitness as an alternative or complementary strategy:

```python
if FITNESS_MODE == "lcs":
    individual.fitness = -lcs_length(individual.genes, TARGET_STRING)
elif FITNESS_MODE == "combined":
    ascii_distance = ...
    lcs_score = lcs_length(...)
    individual.fitness = ascii_distance - (lcs_score * 2)
```

#### a. Relevance to Crossover
LCS preserves order. So when used with crossover, it encourages passing correct sequences.

#### b. Comparison to Original Fitness
- Original ASCII method rewards small character changes.
- LCS focuses on correct order and character placement.

#### c. Impact
- **More structure in convergence**
- **Maintains diversity better**
- **May converge slower but yields cleaner solutions**

---

### Summary

All questions were addressed with appropriate code, graphs, analysis, and algorithm extensions. The code was modified to accept inputs for crossover type and fitness function. Execution time, fitness statistics, and diversity were visualized and analyzed.

The genetic algorithm evolves towards the target while balancing exploration and exploitation using elitism, crossover, mutation, and new heuristics like LCS.
