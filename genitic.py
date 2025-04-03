import time
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # ✅ أضفنا matplotlib للرسم

# الثوابت
GA_POPSIZE = 2048
GA_MAXITER = 1000
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_TARGET = "Hello world!"

# تمثيل الفرد
class GAIndividual:
    def __init__(self, string_val=None):
        self.string = string_val or self.random_string()
        self.fitness = 0

    @staticmethod
    def random_string():
        return ''.join(random.choice(string.printable[:95]) for _ in range(len(GA_TARGET)))

# إنشاء السكان
def init_population():
    return [GAIndividual() for _ in range(GA_POPSIZE)]

# حساب fitness
def calc_fitness(population):
    for individual in population:
        individual.fitness = sum(abs(ord(individual.string[i]) - ord(GA_TARGET[i])) for i in range(len(GA_TARGET)))

# ترتيب حسب الأفضلية
def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)

# نسخ الأفضل (Elitism)
def elitism(population, buffer, esize):
    for i in range(esize):
        buffer[i].string = population[i].string
        buffer[i].fitness = population[i].fitness

# طفرة (Mutation)
def mutate(individual):
    ipos = random.randint(0, len(GA_TARGET) - 1)
    mutated = list(individual.string)
    mutated[ipos] = random.choice(string.printable[:95])
    individual.string = ''.join(mutated)

# التزاوج (Crossover)
def mate(population, buffer):
    esize = int(GA_POPSIZE * GA_ELITRATE)
    elitism(population, buffer, esize)

    for i in range(esize, GA_POPSIZE):
        i1 = random.randint(0, GA_POPSIZE // 2 - 1)
        i2 = random.randint(0, GA_POPSIZE // 2 - 1)
        spos = random.randint(0, len(GA_TARGET) - 1)
        child_str = population[i1].string[:spos] + population[i2].string[spos:]
        child = GAIndividual(child_str)

        if random.random() < GA_MUTATIONRATE:
            mutate(child)

        buffer[i] = child

# تشغيل الخوارزمية
def genetic_algorithm():
    population = init_population()
    buffer = [GAIndividual() for _ in range(GA_POPSIZE)]
    stats = []

    for generation in range(GA_MAXITER):
        start_time = time.time()

        calc_fitness(population)
        sort_by_fitness(population)

        fitness_values = [ind.fitness for ind in population]
        best = fitness_values[0]
        worst = fitness_values[-1]
        avg = np.mean(fitness_values)
        std = np.std(fitness_values)
        elapsed = time.time() - start_time

        stats.append({
            "Generation": generation,
            "Best Fitness": best,
            "Worst Fitness": worst,
            "Average Fitness": avg,
            "Std Dev": std,
            "Elapsed Time (s)": elapsed
        })

        print(f"Gen {generation}: Best={best}, Avg={avg:.2f}, Worst={worst}, Std={std:.2f}")
        print(f"Best string: {population[0].string}")

        if best == 0:
            break

        mate(population, buffer)
        population, buffer = buffer, population

    return pd.DataFrame(stats)

# تشغيل الخوارزمية
df = genetic_algorithm()

# ✅ رسم الجراف المطلوب للبند 3a
plt.figure(figsize=(12, 6))
plt.plot(df['Generation'], df['Best Fitness'], label='Best Fitness', linewidth=2)
plt.plot(df['Generation'], df['Average Fitness'], label='Average Fitness', linestyle='--')
plt.plot(df['Generation'], df['Worst Fitness'], label='Worst Fitness', linestyle=':')

plt.title('Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
