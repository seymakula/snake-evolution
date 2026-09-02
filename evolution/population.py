"""
evolution/population.py — Bir nesli oynat ve fitness'a gore sirala.

Ekransiz. Bu modulde pygame YOK — egitim ekran cizmeden kosuyor,
100 birey x 300 nesil dakikalar aliyor.

obstacles parametresi: engelli ortamda degerlendirme icin. Varsayilan
None, yani mevcut cagrilar (train.py, evolve_live.py) aynen calisir.
"""

from collections import Counter

from agents.neural_agent import NeuralAgent
from evolution.fitness import fitness
from core.game import Game


def play(game, agent):
    """
    Bir oyunu bastan sona oynatir.

    Sozlesme: durum ver, aksiyon al, adim at. Ajanin icinde rastgele
    sayi mi var, matris carpimi mi — oyunun umurunda degil.
    """
    while game.is_alive:
        game.step(agent.act(game.get_state(), game))
    return game


def evaluate_genome(genome, seed, return_game=False, obstacles=None):
    """
    Tek bir genomu degerlendirir: ajan kur, oyun kur, oynat, fitness dondur.

    Genom ile ajan arasindaki kopru burasi — evrim modulu genomlarla
    calisir, ajan nesneleriyle degil.

    return_game=True ise (fitness, oyun, genom) uclusu doner. Egitim
    dongusunde sadece sayiya ihtiyac var; istatistik toplarken oyun da lazim.
    """
    agent = NeuralAgent(genome=genome)
    game = Game(seed=seed, obstacles=obstacles)
    play(game, agent)
    score = fitness(game)

    if return_game:
        return score, game, genome
    return score


def evaluate_population(genomes, seed, obstacles=None):
    """
    Tum nesli degerlendirir, fitness'a gore BUYUKTEN KUCUGE siralar.

    DIKKAT: Nesildeki HERKES ayni tohumu kullanir. Farkli tohum verilseydi
    bir yilan sansli bir yem dizilimi yakalar ve becerisi yuzunden degil
    sansi yuzunden secilirdi. Ayni tohum = herkes ayni sinava girer.

    Siralamanin azalan olmasi onemli: tournament_select min(indeksler)
    ile kazanani buluyor, yani listenin sirali oldugunu VARSAYIYOR.

    Donus: (sirali sonuc listesi, istatistik sozlugu)
    """
    result = []
    for genome in genomes:
        result.append(
            evaluate_genome(genome, seed, return_game=True, obstacles=obstacles)
        )

    result.sort(key=lambda r: r[0], reverse=True)

    n = len(result)
    fitnesses = [r[0] for r in result]
    games = [r[1] for r in result]

    # best_fitness icin max() gerekmiyor — liste zaten sirali.
    stats = {
        "n": n,
        "avg_fitness": sum(fitnesses) / n,
        "best_fitness": fitnesses[0],
        "avg_score": sum(g.score for g in games) / n,
        "best_score": max(g.score for g in games),
        "avg_steps": sum(g.steps for g in games) / n,
        # Olum dagilimi: tek bir fitness sayisi "iyi mi kotu mu" der,
        # bu "neden" der. Projenin en ogretici olcumu.
        "deaths": Counter(g.result.value for g in games),
    }

    return result, stats