"""
render/nn_view.py — Sinir agini CANLI ciz.

Statik semadan farki: noronlarin parlakligi o ANKI aktivasyondan geliyor.
Yilan her adim attiginda harita degisiyor.

Ne gorursun: "onumde tehlike" sensoru yandiginda ondan cikan cizgiler
parlar, bir gizli noron tetiklenir, sonra bir aksiyon secilir.
Sinyalin agin icinde dolasmasi.

Not: agirliklar SABIT (bir oyun boyunca hic degismez). Degisen sey
aktivasyonlar — yani agin ayni beyinle farkli durumlara verdigi tepki.
"""

import numpy as np
import pygame

import config
from core import constants as C

DUYULAR = [
    "tehlike ileri",
    "tehlike sol",
    "tehlike sag",
    "yon yukari",
    "yon asagi",
    "yon sol",
    "yon sag",
    "yem yukari",
    "yem asagi",
    "yem sol",
    "yem sag",
]

AKSIYONLAR = ["duz git", "saga don", "sola don"]

POZITIF = (80, 150, 240)
NEGATIF = (220, 80, 80)
SONUK = (55, 58, 66)
SECILI = (255, 190, 60)


def _karistir(a, b, k):
    """a renginden b rengine k oraninda gecis. k=0 -> a, k=1 -> b."""
    k = max(0.0, min(1.0, k))
    return tuple(int(a[c] + (b[c] - a[c]) * k) for c in range(3))


def _katman_konumlari(n, x, y_ust, y_alt):
    if n == 1:
        return [(x, (y_ust + y_alt) // 2)]
    adim = (y_alt - y_ust) / (n - 1)
    return [(x, int(y_ust + i * adim)) for i in range(n)]


def _akis_ciz(surface, sol, sag, W, kaynak_akt):
    """
    Baglantilari cizer. Parlaklik = |kaynak_aktivasyon x agirlik|.

    Yani sinyal GECEN yollar yanar, gecmeyenler soner. Agirlik buyuk
    olsa bile kaynak noron sessizse o yol karanlik kalir — akisi
    gosteren sey bu.
    """
    akis = np.abs(kaynak_akt)[:, None] * np.abs(W)
    tavan = akis.max()
    if tavan <= 1e-9:
        return

    for i in range(W.shape[0]):
        if abs(kaynak_akt[i]) < 0.05:
            continue  # sessiz noron: hic cizme
        for j in range(W.shape[1]):
            oran = akis[i, j] / tavan
            if oran < 0.12:
                continue  # zayif akis: gorsel kalabalik
            temel = POZITIF if W[i, j] > 0 else NEGATIF
            pygame.draw.line(
                surface,
                _karistir(config.COLOR_BG, temel, 0.2 + 0.8 * oran),
                sol[i],
                sag[j],
                max(1, int(1 + 3 * oran)),
            )


def _noron_ciz(surface, konumlar, aktivasyonlar, temel_renk, yaricap=11, secili=None):
    """Noron parlakligi = |aktivasyon|. Sessiz noronlar sonuk gri."""
    tavan = max(1e-9, np.abs(aktivasyonlar).max())

    for i, (x, y) in enumerate(konumlar):
        oran = abs(aktivasyonlar[i]) / tavan
        renk = _karistir(SONUK, temel_renk, oran)

        pygame.draw.circle(surface, renk, (x, y), yaricap)

        if secili is not None and i == secili:
            pygame.draw.circle(surface, SECILI, (x, y), yaricap + 4, 3)
        else:
            pygame.draw.circle(surface, (30, 34, 40), (x, y), yaricap, 2)


def draw_live_network(surface, net, state, rect, font_small, font_big, baslik=""):
    """
    Agi canli ciz. state = o anki 11 duyu degeri.

    draw_game ile ayni desen: nereye cizecegini kendi bilmiyor,
    dikdortgeni disaridan aliyor.
    """
    girdi_akt, gizli_akt, cikti_akt = net.forward_debug(state)
    secilen = int(np.argmax(cikti_akt))

    sol_x = rect.x + 175
    orta_x = rect.x + rect.width // 2
    sag_x = rect.right - 165
    y_ust = rect.y + 62
    y_alt = rect.bottom - 44

    girdi = _katman_konumlari(config.STATE_SIZE, sol_x, y_ust, y_alt)
    gizli = _katman_konumlari(config.HIDDEN_SIZE, orta_x, y_ust, y_alt)
    cikti = _katman_konumlari(
        C.ACTION_NUM, sag_x, y_ust + (y_alt - y_ust) // 3, y_alt - (y_alt - y_ust) // 3
    )

    _akis_ciz(surface, girdi, gizli, net.w1, girdi_akt)
    _akis_ciz(surface, gizli, cikti, net.w2, gizli_akt)

    _noron_ciz(surface, girdi, girdi_akt, (120, 220, 140))
    _noron_ciz(surface, gizli, gizli_akt, (200, 160, 250))
    _noron_ciz(surface, cikti, cikti_akt, (250, 200, 120), yaricap=14, secili=secilen)

    # --- duyu etiketleri: aktif olanlar parlak ---
    for i, ((x, y), isim) in enumerate(zip(girdi, DUYULAR)):
        aktif = girdi_akt[i] > 0.5
        renk = (150, 230, 170) if aktif else config.COLOR_TEXT_DEAD
        t = font_small.render(isim, True, renk)
        surface.blit(t, (x - 20 - t.get_width(), y - t.get_height() // 2))

    # --- aksiyon etiketleri: secilen vurgulu ---
    for i, ((x, y), isim) in enumerate(zip(cikti, AKSIYONLAR)):
        renk = SECILI if i == secilen else config.COLOR_TEXT_DEAD
        t = font_big.render(f"{isim}  {cikti_akt[i]:+.2f}", True, renk)
        surface.blit(t, (x + 24, y - t.get_height() // 2))

    for x, yazi in ((sol_x, "GIRDI"), (orta_x, "GIZLI"), (sag_x, "CIKTI")):
        t = font_small.render(yazi, True, config.COLOR_TEXT_DEAD)
        surface.blit(t, (x - t.get_width() // 2, y_ust - 32))

    if baslik:
        t = font_big.render(baslik, True, config.COLOR_TEXT)
        surface.blit(t, (rect.x + 14, rect.y + 10))

    alt = font_small.render(
        "parlaklik = o anki aktivasyon   mavi/kirmizi = agirligin isareti   "
        "sari halka = secilen aksiyon",
        True,
        config.COLOR_TEXT_DEAD,
    )
    surface.blit(alt, (rect.x + 14, rect.bottom - 24))
