import numpy as np
import matplotlib.pyplot as plt
import random

POP_SIZE = 50
CHROMOSOME_LEN = 4
MUTATION_RATE = 0.1
GENERATIONS = 100

def get_fitness(chromosome):
    target = np.array([0.08, 0.13, 0.12, 0.05])
    error = np.mean((np.array(chromosome) - target) ** 2)
    fitness = 1.0 / (error + 0.0001)
    return fitness

def create_individual():
    return [random.uniform(0.01, 0.20) for _ in range(CHROMOSOME_LEN)]

def select_parent(population, fitnesses):
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitnesses):
        current += fitness
        if current > pick:
            return individual
    return population[-1]

def crossover(parent1, parent2):
    cut = random.randint(1, CHROMOSOME_LEN - 1)
    child = parent1[:cut] + parent2[cut:]
    return child

def mutate(chromosome):
    for i in range(CHROMOSOME_LEN):
        if random.random() < MUTATION_RATE:
            chromosome[i] += random.uniform(-0.02, 0.02)
            chromosome[i] = max(0.01, min(0.20, chromosome[i]))
    return chromosome

def run_genetic_algorithm():
    population = [create_individual() for _ in range(POP_SIZE)]
    history = []

    for gen in range(GENERATIONS):
        fitnesses = [get_fitness(ind) for ind in population]
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        history.append(best_fitness)
        
        next_generation = []
        for _ in range(POP_SIZE):
            p1 = select_parent(population, fitnesses)
            p2 = select_parent(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child)
            next_generation.append(child)
            
        population = next_generation
        
    final_fitnesses = [get_fitness(ind) for ind in population]
    best_wing = population[np.argmax(final_fitnesses)]
    return best_wing, history

best_wing, history = run_genetic_algorithm()
print(f"Evolved Airfoil Chromosome (Control Point Heights): {np.round(best_wing, 3)}")
