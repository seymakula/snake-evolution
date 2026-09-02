"""
evolve_live.py — Evrimi EKRANDA izle.

main.py'dan farki: orada kaydedilmis tek bir beyin gosteriliyordu,
burada egitimin KENDISI akiyor. Nesil 0'da yilanlar duvara toslar,
40-60 nesil sonra uzayip gezmeye baslarlar.

Numara: populasyon config.POPULATION_SIZE kisilik, ama ekranda sadece
EN IYI 9'u gosteriliyor. 9 bireyle evrim cesitlilik yetersizligi
yuzunden yakinsamaz. Ekrandakiler populasyonun tamami degil, o neslin
vitrini.

Checkpoint: models/checkpoint.npz  (train.py AYRI dosyaya yazar,
boylece burada birkac nesil izlemek oradaki egitimi ezmez.)

Tuslar:
    P       duraklat
    I       cik (kaydeder)
    SPACE   hiz
    N       bu nesli atla
    TAB     gorunum degistir: izgara -> ag -> grafik -> izgara

Proje kokunden: python evolve_live.py [--fresh]
"""

import sys

import numpy as np
import pygame

import config
from core import constants as C
from core.game import Game
from agents.neural_agent import NeuralAgent, NeuralNetwork
from evolution.population import evaluate_population
from evolution.genetic import next_generation
from render import renderer
from render import nn_view
from train import load_checkpoint, save_checkpoint, random_genomes, save_csv
from render import chart_view

# train.py'dan checkpoint fonksiyonlarini aliyoruz. train.py'daki
# if __name__ == "__main__" korumasi sayesinde egitim BASLAMAZ —
# sadece fonksiyonlar import edilir. O satirin ne ise yaradiginin
# somut ornegi.
from train import load_checkpoint, save_checkpoint, random_genomes

CHECKPOINT = "models/checkpoint.npz"

# Nesil bitince ekranda kac saniye beklensin. 0 = aninda gec.
PAUSE_AFTER_GEN = 0.0

# Gorunum modlari
IZGARA, AG, GRAFIK = 0, 1, 2


def build_showcase(results, gen, grid_size):
    """
    O neslin EN IYI 9'undan ekranda oynayacak oyun+ajan ciftleri kurar.

    Bu yilanlar zaten degerlendirildi, skorlari belli. Burada ayni
    genomlari AYNI tohumla tekrar oynatiyoruz ki ekranda gordugun sey
    gercekten fitness'i belirleyen oyun olsun.
    """
    n = grid_size * grid_size
    games, agents = [], []

    for i, (_fit, _game, genome) in enumerate(results[:n]):
        game = Game(seed=gen)  # nesildeki herkes ayni sinava girdi
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
        config.WINDOW_WIDTH,
        config.WINDOW_HEIGHT,
        grid_size,
        config.GAP,
        top_margin=36,
    )

    # ---------- checkpoint ----------
    ck = None
    if "--fresh" not in sys.argv:
        ck = load_checkpoint(CHECKPOINT)

    if ck is not None:
        genomes = ck["genomes"]
        gen = ck["gen"] + 1
        rng = ck["rng"]
        best_ever = ck["best_fitness"]
        history = ck["history"]
        print(f"Checkpoint'ten devam — nesil {gen}")
    else:
        rng = np.random.default_rng(config.TRAIN_SEED)
        genomes = random_genomes(config.POPULATION_SIZE, seed=config.TRAIN_SEED)
        gen = 0
        best_ever = float("-inf")
        history = []
        print("Sifirdan baslaniyor.")

    # Sinir kontrolu ILK degerlendirmeden ONCE — bosuna hesap yapilmasin.
    if gen >= config.GENERATION_LIMIT:
        print(
            f"Zaten {gen} nesil tamamlanmis. "
            f"config.GENERATION_LIMIT'i artir ya da --fresh kullan."
        )
        pygame.quit()
        return

    results, stats = evaluate_population(genomes, seed=gen)
    games, agents = build_showcase(results, gen, grid_size)

    paused = False
    speed_index = 0
    running = True
    step_acc = 0.0
    hold_timer = 0.0
    skip = False
    gorunum = IZGARA

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
                elif event.key == pygame.K_TAB:
                    gorunum = (gorunum + 1) % 3

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
                history.append(
                    [
                        stats["avg_fitness"],
                        stats["best_fitness"],
                        stats["avg_score"],
                        stats["best_score"],
                    ]
                )

                # ISTE EVRIM: yeni nesil uretiliyor.
                genomes = next_generation(results, rng)
                gen += 1

                if gen >= config.GENERATION_LIMIT:
                    print(f"Nesil sinirina ulasildi ({config.GENERATION_LIMIT}).")
                    running = False
                    break

                if gen % 10 == 0:
                    save_checkpoint(genomes, gen, history, rng, best_ever, CHECKPOINT)

                # Degerlendirme SADECE BIR KEZ. Iki kez cagrilirsa nesil
                # gecisleri gereksiz yere iki kat yavaslar.
                results, stats = evaluate_population(genomes, seed=gen)
                games, agents = build_showcase(results, gen, grid_size)
                step_acc = 0.0

        # ---------- cizim ----------
        if gorunum == AG:
            screen.fill(config.COLOR_BG)
            i = renderer.find_leader(games)
            if i is None:
                i = 0

            net = NeuralNetwork(config.STATE_SIZE, config.HIDDEN_SIZE, C.ACTION_NUM)
            net.from_vector(results[i][2])
            nn_view.draw_live_network(
                screen,
                net,
                games[i].get_state(),
                pygame.Rect(0, 40, config.WINDOW_WIDTH, config.WINDOW_HEIGHT - 40),
                renderer._get_font(14),
                renderer._get_font(18),
                baslik=f"Nesil {gen} — #{i+1}",
            )

        elif gorunum == GRAFIK:
            screen.fill(config.COLOR_BG)
            chart_view.draw_chart(
                screen,
                history,
                pygame.Rect(20, 60, config.WINDOW_WIDTH - 40, config.WINDOW_HEIGHT - 120),
                renderer._get_font(14),
                gen=gen,
            )

        else:
            leader = renderer.find_leader(games)
            renderer.draw_grid(screen, games, rects, leader_index=leader)

        _draw_status(screen, gen, speed, paused, stats, best_ever, games)
        pygame.display.set_caption(f"Snake Evolution — CANLI EVRIM — Nesil {gen}")
        pygame.display.flip()

    # Cikista her durumda kaydet: I tusu, pencere kapatma, nesil siniri.
    save_checkpoint(genomes, gen, history, rng, best_ever, CHECKPOINT)
    print(f"Nesil {gen} kaydedildi -> {CHECKPOINT}")

    pygame.quit()
    sys.exit()


def _draw_status(surface, gen, speed, paused, stats, best_ever, games):
    """Ust cubuk. Buradaki sayilar nesiller ilerledikce YUKSELMELI."""
    font = renderer._get_font(17)
    alive = sum(1 for g in games if g.is_alive)

    text = (
        f"Nesil {gen}/{config.GENERATION_LIMIT}   "
        f"pop.ort {stats['avg_fitness']:.2f}   "
        f"pop.en iyi {stats['best_fitness']:.1f}   "
        f"tum zamanlar {best_ever:.1f}   "
        f"skor ort {stats['avg_score']:.1f}   "
        f"{speed}x   canli {alive}/{len(games)}   [TAB]"
    )
    if paused:
        text += "   [DURAKLATILDI]"

    surface.blit(font.render(text, True, config.COLOR_TEXT), (10, 8))


if __name__ == "__main__":
    main()