"""
main.py — Ana döngü.

Buradaki en önemli fikir: ÇİZİM DÖNGÜSÜ ile OYUN DÖNGÜSÜ ayrıdır.
Ekran saniyede 60 kare çizilir; oyun saniyede kaç adım atar, bu
ayrı bir hesaptır (hız çarpanı). İkisi birbirine kilitli olsaydı
ne hızlandırma ne duraklatma ne de ekransız eğitim mümkün olurdu.

Tuşlar:
    P       duraklat / devam
    I       çık
    R       yeni nesil (hepsini sıfırla)
    SPACE   hız çarpanını değiştir (1x -> 2x -> 4x -> 1x)
"""

import sys

import pygame

import config
from core.game import Game
from agents.random_agent import RandomAgent
from render import renderer
import numpy as np
from agents.neural_agent import NeuralAgent

def build_population(grid_size, generation=0):
    genome = np.load("models/best.npy")
    """
    grid_size x grid_size tane BAĞIMSIZ oyun + ajan üretir.

    Tohumlara dikkat: her oyunun ve her ajanın tohumu farklı.
    Aynı olsaydı 9 panelde 9 aynı yılan görürdün — bu, bu projede
    en sık yapılan hatadır.
    """
    n = grid_size * grid_size
    games, agents = [], []

    for i in range(n):
        seed = generation * 1000 + i
        game = Game(seed=seed)
        game.label = f"#{i}"
        games.append(game)
        agents.append(NeuralAgent(genome=genome, name=f"neural-{i}"))

    return games, agents


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Snake Evolution — Nesil 0 (eğitilmiş beyin)")
    clock = pygame.time.Clock()

    grid_size = config.GRID_SIZE
    rects = renderer.compute_panel_rects(
        config.WINDOW_WIDTH, config.WINDOW_HEIGHT, grid_size, config.GAP,top_margin=36
    )

    generation = 0
    games, agents = build_population(grid_size, generation)

    paused = False
    speed_index = 0
    running = True

    # Oyun adımı biriktirici. Her karede hız kadar artar; 1.0'ı geçince
    # bir adım atılır. Bu sayede oyun hızı FPS'ten bağımsız olur.
    step_accumulator = 0.0

    while running:
        # ---------- Olaylar ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # KEYDOWN = basıldığı AN. "basılı mı" kontrolü olsaydı
                # tuşu bir saniye tutunca oyun 60 kez sıfırlanırdı.
                if event.key == pygame.K_i:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r:
                    generation += 1
                    games, agents = build_population(grid_size, generation)
                    pygame.display.set_caption(
                        f"Snake Evolution — Nesil {generation} (eğitilmiş beyinler)"
                    )
                elif event.key == pygame.K_SPACE:
                    speed_index = (speed_index + 1) % len(config.SPEED_MULTIPLIERS)

        speed = config.SPEED_MULTIPLIERS[speed_index]

        # ---------- Oyun mantığı ----------
        # Duraklatıldığında BURASI durur; çizim aşağıda devam eder.
        # Pencere donmaz, oyun ilerlemez. İkisi ayrı olduğu için mümkün.
        if not paused:
            step_accumulator += speed
            while step_accumulator >= 1.0:
                step_accumulator -= 1.0
                for game, agent in zip(games, agents):
                    if game.is_alive:
                        # Sözleşme: durum ver, aksiyon al, adım at.
                        # Sinir ağını taktığında BU SATIRLAR DEĞİŞMEYECEK.
                        action = agent.act(game.get_state(), game)
                        game.step(action)

        # ---------- Çizim ----------
        leader = renderer.find_leader(games)
        renderer.draw_grid(screen, games, rects, leader_index=leader)

        _draw_status(screen, generation, speed, paused, games)

        if all(not g.is_alive for g in games):
            _draw_generation_summary(screen, games)

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()
    sys.exit()


def _draw_status(surface, generation, speed, paused, games):
    """Üst çubuk: nesil, hız, canlı sayısı, en iyi skor."""
    font = renderer._get_font(18)
    alive = sum(1 for g in games if g.is_alive)
    best = max((g.score for g in games), default=0)

    text = f"Nesil: {generation}   Hız: {speed}x   Canlı: {alive}/{len(games)}   En iyi: {best}"
    if paused:
        text += "   [DURAKLATILDI]"

    surface.blit(font.render(text, True, config.COLOR_TEXT), (10, 6))


def _draw_generation_summary(surface, games):
    """
    Nesil bitince skor sıralaması. Üsttekiler "hayatta kaldı",
    alttakiler "elendi".

    ŞİMDİLİK SADECE GÖRSEL — gerçek seçilim yok, çünkü henüz
    seçilecek bir beyin yok (hepsi rastgele). 2. aşamada bu ekran
    gerçekten evolution/ modülüne bağlanacak.
    """
    font = renderer._get_font(20)
    small = renderer._get_font(15)

    w, h = 620, 60 + len(games) * 24
    x = (config.WINDOW_WIDTH - w) // 2
    y = config.WINDOW_HEIGHT - h - 20

    panel = pygame.Surface((w, h))
    panel.set_alpha(235)
    panel.fill((24, 24, 30))
    surface.blit(panel, (x, y))
    pygame.draw.rect(surface, config.COLOR_BORDER, (x, y, w, h), 2)

    surface.blit(
        font.render("NESİL BİTTİ — R ile yenisi", True, config.COLOR_TEXT),
        (x + 16, y + 14),
    )

    ranked = sorted(games, key=lambda g: (g.score, g.steps), reverse=True)
    keep = max(1, len(games) // 3)

    for i, g in enumerate(ranked):
        survived = i < keep
        color = config.COLOR_LEADER if survived else config.COLOR_TEXT_DEAD
        tag = "hayatta kaldı" if survived else "elendi"
        line = f"{i+1}. {g.label}  skor:{g.score}  adım:{g.steps}  {g.result.value}  — {tag}"
        surface.blit(small.render(line, True, color), (x + 16, y + 48 + i * 24))


if __name__ == "__main__":
    main()