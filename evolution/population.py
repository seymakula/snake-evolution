from collections import Counter
from agents.neural_agent import NeuralAgent
from evolution.fitness import fitness
from core.game import Game


def play(game, agent):
    """Bir oyunu baştan sona oynatır. Ekransız — eğitim de böyle koşacak."""
    while game.is_alive:
        game.step(agent.act(game.get_state(), game))
    return game


def evaluate_genome(genome, seed, return_game=False):
    agent = NeuralAgent(genome=genome)
    game = Game(seed=seed)
    play(game, agent)
    score = fitness(game)
    if return_game:
        return score, game, genome
    return score


def evaluate_population(genomes, seed):
    result = []
    for genome in genomes:
        gen = evaluate_genome(genome, seed, return_game=True)
        result.append(gen)
    result.sort(key=lambda r: r[0], reverse=True)

    n = len(result)
    fitnesses = [r[0] for r in result]
    games = [r[1] for r in result]

    stats = {
        "n": n,
        "avg_fitness": sum(fitnesses) / n,
        "best_fitness": fitnesses[0],
        "avg_score": sum(g.score for g in games) / n,
        "best_score": max(g.score for g in games),
        "avg_steps": sum(g.steps for g in games) / n,
        "deaths": Counter(g.result.value for g in games),
    }

    return result, stats

    return result
