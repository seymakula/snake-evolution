import numpy as np
import config
def tournament_select(result, k, rng):
    n=len(result)
    indeksler=rng.integers(0, n, k)
    wins=min(indeksler)
    return result[wins][2]
def crossover(parent_a, parent_b, rng):
    maske=rng.random(len(parent_a)) < 0.5
    return np.where(maske, parent_a, parent_b)

def mutate(genome,rng):
    maske=rng.random(len(genome)) < config.MUTATION_RATE
    noise=rng.normal(0, config.MUTATION_STRENGTH, len(genome))
    return (genome + maske * noise)

def next_generation(results, rng): 
    new_genomes=[]
    for r in results[:config.ELITE_COUNT]:
        new_genomes.append(r[2].copy())
    while len(new_genomes) < config.POPULATION_SIZE:
        parent_a=tournament_select(results, config.TOURNAMENT_SIZE, rng)
        parent_b=tournament_select(results, config.TOURNAMENT_SIZE, rng)
        child=crossover(parent_a, parent_b, rng)
        child=mutate(child, rng)
        new_genomes.append(child)
    return new_genomes