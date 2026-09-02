"""
render/chart_view.py — Canli ogrenme egrisi.

plot_history.py egitim BITTIKTEN sonra matplotlib ile cizer.
Bu modul egitim DEVAM EDERKEN pygame ile cizer: nesil ilerledikce
egri buyur, sicramayi ani aninda gorursun.

Fitness degil SKOR ciziliyor. Fitness soyut bir sayi; skor somut:
kac yem yendi. "Nesil 45'te hicbiri 2 yem yiyemiyordu, simdi 97"
demek daha anlasilir.
"""

import pygame

import config

RENK_ORT = (240, 150, 70)     # populasyon ortalamasi
RENK_BEST = (90, 200, 120)    # en iyi birey
RENK_EKSEN = (70, 74, 84)


def draw_chart(surface, history, rect, font, gen=None):
    """
    history: [avg_fitness, best_fitness, avg_score, best_score] satirlari.
    Skor sutunlari indeks 2 ve 3.

    draw_game ile ayni desen: nereye cizecegini kendi bilmiyor,
    dikdortgeni disaridan aliyor.
    """
    pygame.draw.rect(surface, (22, 24, 28), rect)
    pygame.draw.rect(surface, RENK_EKSEN, rect, 1)

    if len(history) < 2:
        t = font.render("nesil verisi birikiyor...", True, config.COLOR_TEXT_DEAD)
        surface.blit(t, (rect.x + 10, rect.y + 8))
        return

    sol = rect.x + 46
    sag = rect.right - 10
    ust = rect.y + 18
    alt = rect.bottom - 18

    avg = [h[2] for h in history]
    best = [h[3] for h in history]

    tavan = max(max(avg), 1.0)
    n = len(history)

    def nokta(i, deger):
        x = sol + (sag - sol) * (i / max(1, n - 1))
        y = alt - (alt - ust) * (deger / tavan)
        return (int(x), int(y))

    # Yatay kilavuz cizgileri
    for k in (0.5, 1.0):
        y = int(alt - (alt - ust) * k)
        pygame.draw.line(surface, RENK_EKSEN, (sol, y), (sag, y), 1)
        t = font.render(f"{tavan * k:.0f}", True, config.COLOR_TEXT_DEAD)
        surface.blit(t, (rect.x + 6, y - t.get_height() // 2))

    pygame.draw.line(surface, RENK_EKSEN, (sol, ust), (sol, alt), 1)
    pygame.draw.line(surface, RENK_EKSEN, (sol, alt), (sag, alt), 1)

    # Egriler. En iyi birey her zaman ortalamanin ustunde olmali —
    # aradaki bosluk populasyondaki CESITLILIGIN olcusu. Kapanirsa
    # populasyon tek cozume saplanmis demektir.
   
    pygame.draw.lines(surface, RENK_ORT, False,
                      [nokta(i, v) for i, v in enumerate(avg)], 2)

    # Efsane
    bilgi = f"ortalama skor {avg[-1]:.2f}   (tavan {tavan:.1f})"
    t = font.render(bilgi, True, config.COLOR_TEXT_DEAD)
    surface.blit(t, (sag - t.get_width(), rect.y + 3))

    if gen is not None:
        t = font.render(f"nesil 0..{gen}", True, config.COLOR_TEXT_DEAD)
        surface.blit(t, (sol + 4, rect.y + 3))