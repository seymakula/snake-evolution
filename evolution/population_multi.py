"""
evolution/population_multi.py — Cok oyunlu degerlendirme.

SORUN: Mevcut population.py her genomu TEK bir oyunla degerlendiriyor.
Bir genom o ozel yem dizilimine iyi denk geldigi icin yuksek fitness
alabilir. Gercekten iyi mi, sansli mi — ayirt edemiyorsun.
Buna asiri uyum (overfitting) denir.

COZUM: Her genomu N farkli tohumla oynat, fitness'larin ORTALAMASINI al.
Bir genom ancak bircok farkli yem diziliminde iyi oynuyorsa yuksek puan
alir — yani gercekten genel bir strateji ogrenmistir.

DIKKAT: Her bireye BIRER farkli tohum vermek YANLIS olurdu — o zaman
kimin sansli kimin becerikli oldugunu ayirt edemezsin. Herkes ayni
N tohum setiyle sinava girer.

Mevcut population.py'a HIC DOKUNULMADI. Iki versiyon da elinde dursun
ki karsilastirabilesin.
"""

from collections import Counter

import config
from evolution.population import evaluate_genome


def evaluate_genome_multi(genome, seed, n_games=None, return_game=False):
    """
    Ayni genomu n_games farkli tohumla oynatir, fitness ortalamasini doner.

    Tohumlar seed, seed+1, seed+2 ... seklinde. Populasyondaki HERKES
    ayni tohum setini kullanir.
    """
    n_games = n_games or config.GAMES_PER_GENOME

    fitnesses = []
    last_game = None

    for i in range(n_games):
        score, game, _ = evaluate_genome(genome, seed + i, return_game=True)
        fitnesses.append(score)
        last_game = game

    ortalama = sum(fitnesses) / len(fitnesses)

    if return_game:
        # Istatistikler icin son oyunu doner. Yaklasik olur ama
        # trend dogru kalir.
        return ortalama, last_game, genome
    return ortalama


def evaluate_population_multi(genomes, seed, n_games=None):
    """evaluate_population ile ayni sozlesme, sadece cok oyunlu."""
    results = []

    for genome in genomes:
        results.append(
            evaluate_genome_multi(genome, seed, n_games, return_game=True)
        )

    results.sort(key=lambda r: r[0], reverse=True)

    n = len(results)
    fitnesses = [r[0] for r in results]
    games = [r[1] for r in results]

    stats = {
        "n": n,
        "avg_fitness": sum(fitnesses) / n,
        "best_fitness": fitnesses[0],
        "avg_score": sum(g.score for g in games) / n,
        "best_score": max(g.score for g in games),
        "avg_steps": sum(g.steps for g in games) / n,
        "deaths": Counter(g.result.value for g in games),
    }

    return results, stats
