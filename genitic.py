import time
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
sizeOf_population = 2048
max_possib_gens = 1000
eletism_ratio = 0.10
mutationRate = 0.25
the_targetted_String = "loay,mohammad"
limitNoImprove = 100#if we reach this number gens without improvem we stop
TOURNAMENT_K = 5##number of competitors in the tournament selection
STOCHASTIC_P = 0.8##probability of selecting the best individual in the stochastic tournament selection
AGING_MAX_AGE = 10

# here we ask the user to inpput what he like the parameters to be, like for exampke,
#the cross over type and fitness type and parent selection type
#if the input is not true we raise error here
try:
    cross_overTYPE = input("Enter crossover type (SINGLE / TWO / UNIFORM): ").strip().upper() #we get the input and uppercase it and the same we do in others inputs
    assert cross_overTYPE in {"SINGLE", "TWO", "UNIFORM"}


    fitness_mode = input("Choose fitnes type(ascii / lcs / combined): ").strip().lower()
    assert fitness_mode in {"ascii", "lcs", "combined"}
#relevant to question 10 
    selection_metod = input("Enter parent selection method (RWS / SUS / TOURNAMENT / STOCHASTIC_TOURNAMENT): ").strip().upper()
    assert selection_metod in {"RWS", "SUS", "TOURNAMENT", "STOCHASTIC_TOURNAMENT"}

except AssertionError:
    print("Invalid input, Please put  valid option.")
    exit()

#here the basic class which is related to the individual and have the properties of the individual
class Individual:
    def __init__(self, value=None):
        self.genes = value or self._generate_random_genes()#using this we geneerate random string that is the same length as the target string
        self.fitness = 0
        self.age = 0

    @staticmethod
    def _generate_random_genes():
        return ''.join(random.choice(string.printable[:95]) for _ in range(len(the_targetted_String)))

# Population functions
def create_population():
    return [Individual() for _ in range(sizeOf_population)]#here we create individuals as the size of the maxumim population

# Longest Common Subsequence this is related to question 7.
def lcs_length(a, b):#we calcualte the longest common sequence between two strings
    ## Create a 2D grid where each cell [i][j] will store the LCS length
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[len(a)][len(b)]

#assign fitness values to the individuals, the lower the better, and thies is calculated based on the the fitness mode, 
#whether it is ascii or lcs or combined
def evaluate_population(population):
    for individual in population:
        if fitness_mode == "ascii":
            individual.fitness = sum(abs(ord(individual.genes[i]) - ord(the_targetted_String[i])) for i in range(len(the_targetted_String)))
        elif fitness_mode == "lcs":
            individual.fitness = -lcs_length(individual.genes, the_targetted_String)
        elif fitness_mode == "combined":
            ascii_distance = sum(abs(ord(individual.genes[i]) - ord(the_targetted_String[i])) for i in range(len(the_targetted_String)))
            lcs_score = lcs_length(individual.genes, the_targetted_String)
            individual.fitness = ascii_distance - (lcs_score * 2)
        else:
            raise ValueError("Invalid .. Choose ascii, lcs, or combined.")

#Measures how different the individuals are, on average.
    #We compare every pair and count how many characters differ
def average_hamming_distance(population):
    total_dif = 0
    count = 0
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            dist = sum(a != b for a, b in zip(population[i].genes, population[j].genes))        #counting different chrom for pairs
            
            total_dif += dist
            count += 1      #instead of pre calculating the total number of combinations (n choose 2) we count each comparison indivisually 
    return total_dif / count
#here we count how many uinique characters appear at each position in the population
#higher mean more diversity
def get_total_unique_alleles(population):
    gene_len = len(population[0].genes) ##assume all individual have the same length
    total_unique = 0 #
    for position in range(gene_len):##for each position in the gene we count how many unique characters appear at that position
        alleles = set(ind.genes[position] for ind in population)     #counting unique chromosomes in on the same index
        total_unique += len(alleles)
    return total_unique
#this function measures how diverse the population is using Shannon 
#highter entropy means more diversity and lower entropy means less diversity
def get_avg_entropy(population):
    gene_length = len(population[0].genes)##assume all individual have the same length
    theTotal_entrop = 0 ##total entropy for all positions

    for position in range(gene_length): 
        allele_counts = {}  


        for individual in population:       #counting frequency of each char at position i
            allele = individual.genes[position]
            if allele not in allele_counts:#if the allele isnt in the dictionary we add it and set the value to 0
                allele_counts[allele] = 0

            allele_counts[allele] += 1

        # finding the probability for each char
        probabilities = [count / sizeOf_population for count in allele_counts.values()]

        # compute entropy on this position
        entropy_here = 0
        for p in probabilities:
            if p > 0:
                entropy_here -= p * np.log2(p)

        theTotal_entrop += entropy_here

    # average entropy over all positions
    avg_entropy = theTotal_entrop / gene_length
    return avg_entropy

def roulette_wheel_selection(population, cumulative_scaled, total_scaled): #we pre-calculate the cumulative scaled and total scaled at thr beginig of each generation to prevent redundant calculations
    pick = random.uniform(0, total_scaled)
    for i, cumulative in enumerate(cumulative_scaled): # we iterate through the cumulative scaled until we reach to a cumulative index where pick become larger
        if pick <= cumulative:
            return population[i]

def pick_parents_sus(population, scaled_fitness, total_scaled):
    """
    Selects 2 parents using Stochastic Universal Sampling
    We spread two equally spaced pointers over the cumulative fitness space.
    """

    # Distance between selection pointers (we need 2 parents)
    spacing = total_scaled / 2

    # Random starting point between 0 and the spacing distance
    start = random.uniform(0, spacing)

    # Generate the two selection points
    pointers = [start + i * spacing for i in range(2)]

    selected_parents = []
    cumulative_sum = 0
    pointer_index = 0

    # Walk through the fitness values and check when a pointer "lands" inside a region
    for i, fitness_value in enumerate(scaled_fitness):
        cumulative_sum += fitness_value

        # While current pointer falls within the current cumulative range
        while pointer_index < len(pointers) and cumulative_sum >= pointers[pointer_index]:
            selected_parents.append(population[i])
            pointer_index += 1

    return selected_parents


def tournament_selection(population, k):#here we select k random individuals and sort them by fitness, then we return the best one
    competitors = random.sample(population, k)
    competitors.sort(key=lambda x: x.fitness)##sort the competitors by fitness
    return competitors[0]##return the best

##here we select k random individuals and sort them by fitness,
# # then we return the best one with probability p and a random one from the rest with probability 1-p
def stochastic_tournament_selection(population, k, p):
    competitors = random.sample(population, k)
    competitors.sort(key=lambda x: x.fitness)
    if random.random() < p:
        return competitors[0]
    else:
        return random.choice(competitors[1:])##return a random individual from the rest

def select_parents(population, scaled_fitness=None, cumulative_scaled=None, total_scaled=None):##this function is used to select the parents based on the selection method
    if selection_metod == "RWS":
        return roulette_wheel_selection(population, cumulative_scaled, total_scaled)
    elif selection_metod == "TOURNAMENT":
        return tournament_selection(population, TOURNAMENT_K)
    elif selection_metod == "STOCHASTIC_TOURNAMENT":
        return stochastic_tournament_selection(population, TOURNAMENT_K, STOCHASTIC_P)
    else:
        raise ValueError("Invalid selection method or SUS should be handled in crossover()")

def sort_population(population):##sort the population by fitness, the lower the better
    population.sort(key=lambda x: x.fitness)


#this function is used to apply elitism, we take the best individuals from the current population and add them to the next generation
def apply_elitism(current_population, next_generation, elite_size):##
    for i in range(elite_size):
        next_generation[i].genes = current_population[i].genes
        next_generation[i].fitness = current_population[i].fitness

#this function is used to mutate the individual, we randomly select a character in the gene and replace it with a random character from the printable characters
#this help with the exploration
def mutate(individual):
    index = random.randint(0, len(the_targetted_String) - 1)
    genes = list(individual.genes)
    genes[index] = random.choice(string.printable[:95])
    individual.genes = ''.join(genes)

##this function is used to crossover the individuals, we select two parents and create a child from them
def crossover(current_population, next_generation, scaled_fitness=None, cumulative_scaled=None, total_scaled=None):
    elite_size = int(sizeOf_population * eletism_ratio)
    apply_elitism(current_population, next_generation, elite_size)

    for i in range(elite_size, sizeOf_population):
        if selection_metod == "SUS":
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

        if cross_overTYPE == "SINGLE":
            point = random.randint(1, len(the_targetted_String) - 1)
            child_genes = genes1[:point] + genes2[point:]
        elif cross_overTYPE == "TWO":
            p1 = random.randint(0, len(the_targetted_String) - 2)
            p2 = random.randint(p1 + 1, len(the_targetted_String) - 1)
            child_genes = genes1[:p1] + genes2[p1:p2] + genes1[p2:]
        elif cross_overTYPE == "UNIFORM":
            child_genes = ''.join(random.choice([g1, g2]) for g1, g2 in zip(genes1, genes2))
        else:
            raise ValueError("error, Choose SINGLE, TWO, or UNIFORM.")

        child = Individual(child_genes)
        if random.random() < mutationRate:
            mutate(child)##here we mutate the child with a probability of mutationRate

        next_generation[i] = child##here we add the child to the next generation

# Main algorithm
def run_genetic_algorithm():##this function is used to run the genetic algorithm, we create the population and then we run the algorithm for max_possib_gens generations
    current_generation = create_population()
    next_generation = [Individual() for _ in range(sizeOf_population)]##here we create the next generation

    stats_log = []
    fitnes_log = []

    best_fitness = None##best fitness of the current generation starting from None
    no_improvement_counter = 0 ## Counter for generations without improvement

    for gen in range(max_possib_gens):
        start_time = time.time()
        start_ticks = time.process_time()

        evaluate_population(current_generation)##here we evaluate the population and assign fitness values to the individuals
        sort_population(current_generation)##here we sort the population by fitness, the lower the better

        #avg_hamming = average_hamming_distance(current_generation)
        unique_alleles = get_total_unique_alleles(current_generation)##here we count how many unique characters appear at each position in the population
        entropy= get_avg_entropy(current_generation)##here we calculate the entropy of the population

        for ind in current_generation:
            ind.age += 1

        fitnes_Values = [ind.fitness for ind in current_generation]##here we get the fitness values of the current generation
        fitnes_log.append(fitnes_Values.copy())##here we append the fitness values to the fitness log

        if fitness_mode == "lcs" or "combined":##here we check if the fitness mode is lcs or combined, if it is we need to make sure that the fitness values are positive
            min_fitness = min(fitnes_Values)##here we get the minimum fitness value
            fitnes_Values = [f - min_fitness + 1 for f in fitnes_Values]
        
        # Precompute scaled fitness and cumulative sums for RWS/SUS once per generation)
        scaled_fitness = []##here we create the scaled fitness list
        cumulative_scaled = []#create the cumulative scaled list
        total_scaled = 0

        if selection_metod in {"RWS", "SUS"}:
            scaled_fitness = [1 / (f + 1) for f in fitnes_Values]
            total_scaled = sum(scaled_fitness)
            cumulative_scaled = np.cumsum(scaled_fitness).tolist()

        best = fitnes_Values[0]##here we get the best fitness value of the current generation which is always the first one after sorting
        worst = fitnes_Values[-1]
        avg = np.mean(fitnes_Values)
        standart_dev = np.std(fitnes_Values)                  
        
        variance = np.var(fitnes_Values)                 
        elapsed_time = time.time() - start_time
        end_ticks = time.process_time()
        clock_ticks = end_ticks - start_ticks

        #Top-Average Selection Probability Ratio
        top_n = int(sizeOf_population * eletism_ratio)
        top_avg = np.mean(fitnes_Values[:top_n])
        top_avg_ratio = top_avg / avg 

        stats_log.append({
            "Generation": gen,
            "Best Fitness": best,
            "Worst Fitness": worst,
            "Average Fitness": avg,
            "Std Dev": standart_dev,
            "Fitness Variance": variance,
            "Top-Average Ratio": top_avg_ratio,
            #"Avg Hamming Distance": avg_hamming,
            "Unique Alleles": unique_alleles,
            "Shannon Entropy": entropy,
            "Elapsed Time (s)": elapsed_time,
            "Clock Ticks": clock_ticks
        })

        print(
            f"geniration {gen}: "
            f"best={best}, Avg={avg:.2f}, Worst={worst}, "
            f"Standart dDev={standart_dev:.2f}, Variance={variance:.2f}, "
            f"Top-Avg Ratio={top_avg_ratio:.3f}, "
            f"UniqueAlleles={unique_alleles}, Entropy={entropy:.3f}, "
            f"Elapsed={elapsed_time:.5f}s"
        )
        print(f"Best matched: {current_generation[0].genes}")##here we print the best matched string

        if current_generation[0].genes == the_targetted_String:
            print("Target string reached, Stopping early.  ")
            break

        if best_fitness is None or best < best_fitness:
            best_fitness = best
            
            no_improvement_counter = 0
        
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= limitNoImprove:##here we check if the number of geni0rations without improvement is greater than the limit, if it is we stop the algorithm
                print(f"No improvement in {limitNoImprove} generations . stopping early...")
                break
#here we apply crossover to the current generation and create the next generation
        crossover(current_generation, next_generation, scaled_fitness, cumulative_scaled, total_scaled)#
        current_generation, next_generation = next_generation, current_generation

    return pd.DataFrame(stats_log), fitnes_log


# Run and plot results
results_df, fitness_over_time = run_genetic_algorithm()##call the genitic algorithm

plt.figure(figsize=(12, 6))## Plotting the fitness progression over generations
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
# Plotting the fitness distribution using boxplot
plt.figure(figsize=(14, 6))
plt.boxplot(fitness_over_time, vert=True, patch_artist=True, showfliers=True)
plt.title('Fitness Distribution Per Generation (Boxplot)')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.grid(True)
plt.tight_layout()
plt.show()
