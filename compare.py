"""
compare.py — TRANSFER OGRENME deneyi.

Iki populasyon, ikisi de ENGELLI haritada evrimlesiyor:

    SOL : models/best_10x10_engelsiz.npy'den mutasyonla uretildi.
          Yani 300 nesil ENGELSIZ tahtada ogrenmis bir beynin cocuklari.
    SAG : tamamen rastgele. Hicbir sey bilmiyor.

Olculen soru: eski ortamda ogrenilen sey yeni ortamda ise yariyor mu?

On olcum (tek oyun, evrimsiz):
    egitilmis model engelsiz tahtada -> skor 20
    ayni model engelli tahtada       -> skor  2
Yani hayatta kalma transfer oluyor (engellere carpmiyor), yem
toplama olmuyor. Ogrendigi "bos tahtada sistematik tarama" stratejisi
engellerle bozuluyor.

Beklenti: sol taraf basta onde olur (duvardan kacmayi biliyor), ama
sag taraf yakalayabilir — cunku o bastan engelli ortama uyum sagliyor,
eski aliskanliklari yok. Olursa buna NEGATIF TRANSFER denir.

Tuslar:
    TAB     gorunum: yilanlar <-> grafik
    P       duraklat
    N       nesli atla
    SPACE   hiz
    I       cik

Proje kokunden: python compare.py
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

MODEL_YOLU = "models/best_10x10_engelsiz.npy"

# Populasyon: her taraf icin ayri. Her nesilde 2 x POP oyun degerlendirilecek,
# o yuzden config.POPULATION_SIZE'dan kucuk tutmak izlenebilirligi artirir.
POP = 100

GRID = 2            # her tarafta 2x2 = 4 panel, toplam 8 yilan
PAUSE_AFTER_GEN = 0.0

# Egitilmis populasyonu tek genomdan uretirken kullanilan mutasyon.
# config.MUTATION_STRENGTH'ten buyuk — 100 ozdes kopya yerine
# cesitlilik lazim, yoksa secilim yapacak malzeme olmaz.
TOHUM_MUTASYON = 0.35

YILANLAR, GRAFIK = 0, 1

RENK_SOL = (110, 200, 250)
RENK_SAG = (250, 170, 90)


def genome_size():
    return (
        config.STATE_SIZE * config.HIDDEN_SIZE
        + config.HIDDEN_SIZE
        + config.HIDDEN_SIZE * C.ACTION_NUM
        + C.ACTION_NUM
    )


def egitilmis_populasyon(n, rng):
    """
    Tek bir egitilmis genomdan n kisilik populasyon uretir.

    Ilk birey ORIJINALIN AYNISI (elitizm mantigi — en iyiyi kaybetme),
    gerisi mutasyonlu kopyalar. Hepsi ozdes olsaydi caprazlama hicbir
    sey uretmezdi ve evrim baslayamazdi.
    """
    taban = np.load(MODEL_YOLU)
    pop = [taban.copy()]
    for _ in range(n - 1):
        pop.append(taban + rng.normal(0, TOHUM_MUTASYON, len(taban)))
    return pop


def rastgele_populasyon(n, rng):
    size = genome_size()
    return [rng.normal(0, 1, size) for _ in range(n)]


def showcase(results, gen, k):
    """O neslin en iyi k bireyinden ekranda oynayacak oyun+ajan ciftleri."""
    oyunlar, ajanlar = [], []
    for i, (_fit, _game, genom) in enumerate(results[:k]):
        oyun = Game(seed=gen, obstacles=config.OBSTACLES)
        oyun.label = f"#{i+1}"
        oyunlar.append(oyun)
        ajanlar.append(NeuralAgent(genome=genom))
    return oyunlar, ajanlar


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    yari = config.WINDOW_WIDTH // 2
    ust = 78

    # Her yari icin ayri panel izgarasi. compute_panel_rects zaten
    # "sana verilen alani bol" diyor — iki kez cagirip kaydiriyoruz.
    sol_rects = renderer.compute_panel_rects(
        yari - 16, config.WINDOW_HEIGHT - ust - 16, GRID, config.GAP
    )
    sag_rects = renderer.compute_panel_rects(
        yari - 16, config.WINDOW_HEIGHT - ust - 16, GRID, config.GAP
    )
    sol_rects = [r.move(8, ust) for r in sol_rects]
    sag_rects = [r.move(yari + 8, ust) for r in sag_rects]

    rng_sol = np.random.default_rng(config.TRAIN_SEED)
    rng_sag = np.random.default_rng(config.TRAIN_SEED + 1)

    genom_sol = egitilmis_populasyon(POP, rng_sol)
    genom_sag = rastgele_populasyon(POP, rng_sag)

    gen = 0
    hist_sol, hist_sag = [], []

    res_sol, st_sol = evaluate_population(genom_sol, seed=gen,
                                          obstacles=config.OBSTACLES)
    res_sag, st_sag = evaluate_population(genom_sag, seed=gen,
                                          obstacles=config.OBSTACLES)
    oyun_sol, ajan_sol = showcase(res_sol, gen, GRID * GRID)
    oyun_sag, ajan_sag = showcase(res_sag, gen, GRID * GRID)

    paused = False
    speed_index = 0
    running = True
    step_acc = 0.0
    hold = 0.0
    skip = False
    gorunum = YILANLAR

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
                    gorunum = (gorunum + 1) % 2

        speed = config.SPEED_MULTIPLIERS[speed_index]
        canli = any(o.is_alive for o in oyun_sol + oyun_sag)

        # ---------- oyun mantigi ----------
        if not paused and canli and not skip:
            step_acc += speed
            while step_acc >= 1.0:
                step_acc -= 1.0
                for oyun, ajan in zip(oyun_sol + oyun_sag, ajan_sol + ajan_sag):
                    if oyun.is_alive:
                        oyun.step(ajan.act(oyun.get_state(), oyun))

        # ---------- nesil bitti ----------
        if (not canli or skip) and not paused:
            hold += dt
            if hold >= PAUSE_AFTER_GEN or skip:
                hold = 0.0
                skip = False

                hist_sol.append(st_sol["avg_score"])
                hist_sag.append(st_sag["avg_score"])

                # ISTE EVRIM — iki populasyon AYRI AYRI evrimlesiyor.
                genom_sol = next_generation(res_sol, rng_sol)
                genom_sag = next_generation(res_sag, rng_sag)
                gen += 1

                # Ayni tohum: iki taraf ayni sinava girsin. Farkli
                # tohum verilseydi skor farkinin beyinden mi sanstan mi
                # geldigini ayirt edemezdik.
                res_sol, st_sol = evaluate_population(
                    genom_sol, seed=gen, obstacles=config.OBSTACLES)
                res_sag, st_sag = evaluate_population(
                    genom_sag, seed=gen, obstacles=config.OBSTACLES)

                oyun_sol, ajan_sol = showcase(res_sol, gen, GRID * GRID)
                oyun_sag, ajan_sag = showcase(res_sag, gen, GRID * GRID)
                step_acc = 0.0

        # ---------- cizim ----------
        screen.fill(config.COLOR_BG)

        if gorunum == YILANLAR:
            for oyun, rect in zip(oyun_sol, sol_rects):
                renderer.draw_game(screen, oyun, rect)
            for oyun, rect in zip(oyun_sag, sag_rects):
                renderer.draw_game(screen, oyun, rect)
        else:
            h = config.WINDOW_HEIGHT - ust - 30
            _grafik(screen, hist_sol, pygame.Rect(14, ust, yari - 28, h),
                    RENK_SOL, "EGITILMIS soyu", hist_sol + hist_sag)
            _grafik(screen, hist_sag, pygame.Rect(yari + 14, ust, yari - 28, h),
                    RENK_SAG, "SIFIRDAN", hist_sol + hist_sag)

        _ustbilgi(screen, gen, speed, paused, st_sol, st_sag, yari)
        pygame.display.set_caption(
            f"Snake Evolution — TRANSFER DENEYI — Nesil {gen}")
        pygame.display.flip()

    pygame.quit()
    sys.exit()


def _ustbilgi(surface, gen, speed, paused, st_sol, st_sag, yari):
    font = renderer._get_font(16)
    kucuk = renderer._get_font(13)

    bas = f"ENGELLI HARITA   Nesil {gen}   {speed}x   [TAB] gorunum  [N] atla"
    if paused:
        bas += "   [DURAKLATILDI]"
    surface.blit(font.render(bas, True, config.COLOR_TEXT), (12, 8))

    sol = f"EGITILMIS   ort {st_sol['avg_score']:.1f}   en iyi {st_sol['best_score']}"
    sag = f"SIFIRDAN   ort {st_sag['avg_score']:.1f}   en iyi {st_sag['best_score']}"
    surface.blit(font.render(sol, True, RENK_SOL), (14, 32))
    surface.blit(font.render(sag, True, RENK_SAG), (yari + 14, 32))

    surface.blit(kucuk.render("300 nesil engelsiz tahtada egitildi",
                              True, config.COLOR_TEXT_DEAD), (14, 54))
    surface.blit(kucuk.render("rastgele agirliklar",
                              True, config.COLOR_TEXT_DEAD), (yari + 14, 54))


def _grafik(surface, seri, rect, renk, baslik, ortak):
    """
    Nesil-skor egrisi.

    ortak: IKI serinin birlesimi. Tavan ondan hesaplaniyor ki iki grafik
    AYNI olcekte olsun — ayri olceklenselerdi biri 2'ye biri 50'ye gore
    cizilir ve karsilastirma yanilticiliga donerdi.
    """
    font = renderer._get_font(13)
    pygame.draw.rect(surface, (22, 24, 28), rect)
    pygame.draw.rect(surface, (70, 74, 84), rect, 1)

    surface.blit(font.render(baslik, True, renk), (rect.x + 8, rect.y + 6))

    if len(seri) < 2:
        surface.blit(font.render("nesil verisi birikiyor...", True,
                                 config.COLOR_TEXT_DEAD),
                     (rect.x + 8, rect.y + 26))
        return

    tavan = max(max(ortak), 1.0)
    sol_x, sag_x = rect.x + 44, rect.right - 12
    ust_y, alt_y = rect.y + 30, rect.bottom - 22
    n = len(seri)

    for k in (0.5, 1.0):
        y = int(alt_y - (alt_y - ust_y) * k)
        pygame.draw.line(surface, (70, 74, 84), (sol_x, y), (sag_x, y), 1)
        t = font.render(f"{tavan * k:.0f}", True, config.COLOR_TEXT_DEAD)
        surface.blit(t, (rect.x + 8, y - t.get_height() // 2))

    pygame.draw.line(surface, (70, 74, 84), (sol_x, ust_y), (sol_x, alt_y), 1)
    pygame.draw.line(surface, (70, 74, 84), (sol_x, alt_y), (sag_x, alt_y), 1)

    noktalar = [
        (int(sol_x + (sag_x - sol_x) * (i / max(1, n - 1))),
         int(alt_y - (alt_y - ust_y) * (v / tavan)))
        for i, v in enumerate(seri)
    ]
    pygame.draw.lines(surface, renk, False, noktalar, 2)

    son = font.render(f"son {seri[-1]:.2f}   tavan {tavan:.1f}",
                      True, config.COLOR_TEXT_DEAD)
    surface.blit(son, (sag_x - son.get_width(), rect.y + 6))


if __name__ == "__main__":
    main()