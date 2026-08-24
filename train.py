import config 
import numpy as np
from core import constants as C
from evolution.population import evaluate_population
from evolution.genetic import next_generation

def random_genomes(n, seed=0):
        """n tane rastgele genom uret. Nesil 0'in ham maddesi."""
        rng = np.random.default_rng(seed)
        size = (
            config.STATE_SIZE * config.HIDDEN_SIZE
            + config.HIDDEN_SIZE
            + config.HIDDEN_SIZE * C.ACTION_NUM
            + C.ACTION_NUM
        )
        return [rng.normal(0, 1, size) for _ in range(n)]

def main():
    n=config.POPULATION_SIZE
    rng = np.random.default_rng(config.TRAIN_SEED)
    genomes = random_genomes(config.POPULATION_SIZE, seed=config.TRAIN_SEED)
    best_fitness = float("-inf")
    history=[]
    

    for gen in range(config.GENERATION_LIMIT):
        results, stats = evaluate_population(genomes, seed=gen)
        print(
            f"Nesil {gen:>3} | "
            f"fitness ort {stats['avg_fitness']:>6.3f} en iyi {stats['best_fitness']:>7.3f} | "
            f"skor ort {stats['avg_score']:>5.2f} en iyi {stats['best_score']:>2} | "
            f"adım ort {stats['avg_steps']:>5.1f}"
        )

        if gen % 10 == 0:
            deaths = "  ".join(
                f"{name}:{count}" for name, count in stats["deaths"].most_common()
            )
            print(f"          ölümler: {deaths}")
        if stats["best_fitness"] > best_fitness:
            best_fitness = stats["best_fitness"]
            np.save("models/history.npy", np.array([
    [h["avg_fitness"], h["best_fitness"], h["avg_score"], h["best_score"]]
    for h in history
]))
        history.append(stats)
        genomes = next_generation(results, rng)
    print()
    print("=" * 60)
    print(f"Eğitim bitti — {config.GENERATION_LIMIT} nesil")
    print(f"En iyi fitness    : {best_fitness:.3f}")
    print(f"İlk nesil ortalama: {history[0]['avg_fitness']:.3f}")
    print(f"Son nesil ortalama: {history[-1]['avg_fitness']:.3f}")
    print(f"En iyi skor       : {max(h['best_score'] for h in history)}")
    print(f"Kaydedildi        : models/best.npy")
    print("=" * 60)        



if __name__ == "__main__":
    main()