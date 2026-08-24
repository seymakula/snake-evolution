"""
evolve_live.py — Evrimi EKRANDA izle.

main.py'dan farki: orada kaydedilmis tek bir beyin gosteriliyordu,
burada egitimin KENDISI akiyor. Nesil 0'da yilanlar duvara toslar,
40-60 nesil sonra uzayip gezmeye baslarlar.

Numara: populasyon config.POPULATION_SIZE kisilik (100), ama ekranda
sadece EN IYI 9'u gosteriliyor. 9 bireyle evrim cesitlilik yetersizligi
yuzunden yakinsamaz; 100 ile calisir. Ekrandakiler populasyonun tamami
degil, o neslin vitrini.

Tuslar:
    P       duraklat
    I       cik
    SPACE   hiz (1x -> 2x -> 4x)
    N       bu nesli atla, hemen sonrakine gec

Proje kokunden: python evolve_live.py
"""

import sys

import numpy as np
import pygame

import config
from core import constants as C
from core.game import Game
from agents.neural_agent import NeuralAgent
from evolution.population import evaluate_population
from evolution.genetic import next_generation
from render import renderer

# Nesil bitince ekranda kac saniye beklensin (ozet okunabilsin diye)
PAUSE_AFTER_GEN = 1.2


def genome_size():
    return (
        config.STATE_SIZE * config.HIDDEN_SIZE
        + config.HIDDEN_SIZE
        + config.HIDDEN_SIZE * C.ACTION_NUM
        + C.ACTION_NUM
    )


def random_genomes(n, rng):
    size = genome_size()
    return [rng.normal(0, 1, size) for _ in range(n)]


def build_showcase(genomes, results, gen, grid_size):
    """
    O neslin EN IYI 9'undan ekranda oynayacak oyun+ajan ciftleri kurar.

    Not: bu yilanlar zaten degerlendirildi, skorlari belli. Burada
    ayni genomlari AYNI tohumla tekrar oynatiyoruz ki ekranda gordugun
    sey gercekten fitness'i belirleyen oyun olsun.
    """
    n = grid_size * grid_size
    games, agents = [], []

    for i, (fit, _game, genome) in enumerate(results[:n]):
        game = Game(seed=gen)          # nesildeki herkes ayni sinava girdi
        game.label = f"#{i+1}"
        games.append(game)
        agents.append(NeuralAgent(genome=genome))

    return games, agents


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    grid_size = config.GRID_SIZE
    rects = renderer.compute_panel_rects(
        config.WINDOW_WIDTH, config.WINDOW_HEIGHT, grid_size, config.GAP,
        top_margin=36,
    )

    rng = np.random.default_rng(config.TRAIN_SEED)
    genomes = random_genomes(config.POPULATION_SIZE, rng)

    gen = 0
    best_ever = float("-inf")

    paused = False
    speed_index = 0
    running = True
    step_acc = 0.0
    hold_timer = 0.0        # nesil bitince bekleme sayaci
    skip = False

    # --- ilk nesli degerlendir ---
    results, stats = evaluate_population(genomes, seed=gen)
    games, agents = build_showcase(genomes, results, gen, grid_size)

    while running:
        dt = clock.tick(config.FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_SPACE:
                    speed_index = (speed_index + 1) % len(config.SPEED_MULTIPLIERS)
                elif event.key == pygame.K_n:
                    skip = True

        speed = config.SPEED_MULTIPLIERS[speed_index]
        alive = any(g.is_alive for g in games)

        # ---------- oyun mantigi ----------
        if not paused and alive and not skip:
            step_acc += speed
            while step_acc >= 1.0:
                step_acc -= 1.0
                for game, agent in zip(games, agents):
                    if game.is_alive:
                        game.step(agent.act(game.get_state(), game))

        # ---------- nesil bitti mi ----------
        if (not alive or skip) and not paused:
            hold_timer += dt
            if hold_timer >= PAUSE_AFTER_GEN or skip:
                hold_timer = 0.0
                skip = False

                best_ever = max(best_ever, stats["best_fitness"])

                # ISTE EVRIM: yeni nesil uretiliyor.
                genomes = next_generation(results, rng)
                gen += 1

                results, stats = evaluate_population(genomes, seed=gen)
                games, agents = build_showcase(genomes, results, gen, grid_size)
                step_acc = 0.0

        # ---------- cizim ----------
        leader = renderer.find_leader(games)
        renderer.draw_grid(screen, games, rects, leader_index=leader)
        _draw_status(screen, gen, speed, paused, stats, best_ever, games)

        pygame.display.set_caption(
            f"Snake Evolution — CANLI EVRIM — Nesil {gen}"
        )
        pygame.display.flip()

    pygame.quit()
    sys.exit()


def _draw_status(surface, gen, speed, paused, stats, best_ever, games):
    """Ust cubuk. Buradaki sayilar nesiller ilerledikce YUKSELMELI."""
    font = renderer._get_font(17)
    alive = sum(1 for g in games if g.is_alive)

    text = (
        f"Nesil {gen}   "
        f"pop.ort {stats['avg_fitness']:.2f}   "
        f"pop.en iyi {stats['best_fitness']:.1f}   "
        f"tum zamanlar {best_ever:.1f}   "
        f"skor ort {stats['avg_score']:.1f}   "
        f"{speed}x   canli {alive}/{len(games)}"
    )
    if paused:
        text += "   [DURAKLATILDI]"

    surface.blit(font.render(text, True, config.COLOR_TEXT), (10, 8))


if __name__ == "__main__":
    main()
