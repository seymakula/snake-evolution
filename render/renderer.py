"""
render/renderer.py — Oyunu ekrana çizer.

TEK KURAL: Bu dosya oyunu OKUR, asla DEĞİŞTİRMEZ.
Buradaki hiçbir fonksiyon game.step() çağırmaz, skoru değiştirmez.
Çizim tarafı bozulsa bile motor sağlam kalmalı.

Ana fikir: draw_game(surface, game, rect) — "şu oyunu, şu dikdörtgenin
içine çiz". Nereye çizeceğini kendi bilmiyor, dışarıdan alıyor.
Bu sayede 1 panel de 9 panel de aynı fonksiyonla çizilir.
"""

import pygame

import config
from core import constants as C


# --------------------------------------------------------------------
# Izgara geometrisi
# --------------------------------------------------------------------

def compute_panel_rects(window_width, window_height, grid_size, gap, top_margin=0):
    rects = []
    usable_h = window_height - top_margin
    panel_w = (window_width - gap * (grid_size + 1)) // grid_size
    panel_h = (usable_h - gap * (grid_size + 1)) // grid_size

    for row in range(grid_size):
        for col in range(grid_size):
            x = gap + col * (panel_w + gap)
            y = top_margin + gap + row * (panel_h + gap)
            rects.append(pygame.Rect(x, y, panel_w, panel_h))
    return rects

def _cell_size(game, rect):
    """
    Hücre boyutu SABİT DEĞİL — panelin boyutundan hesaplanır.
    Bu yüzden aynı oyun hem küçük panelde hem tam ekranda çizilebiliyor.
    """
    return min(rect.width // game.cols, rect.height // game.rows)


# --------------------------------------------------------------------
# Tek bir oyunu çizme
# --------------------------------------------------------------------

def draw_game(surface, game, rect, show_header=True, is_leader=False):
    header_h = 40 if show_header else 0

    cell = min(rect.width // game.cols, (rect.height - header_h) // game.rows)
    board_w = cell * game.cols
    board_h = cell * game.rows

    ox = rect.x + (rect.width - board_w) // 2
    oy = rect.y + header_h + (rect.height - header_h - board_h) // 2
    dead = not game.is_alive

    # Ölmüş oyunlar grileşir — kimin elendiği bir bakışta görünsün.
    bg = config.COLOR_DEAD_BG if dead else config.COLOR_BOARD
    pygame.draw.rect(surface, bg, (ox, oy, board_w, board_h))

    # Yem
    if game.food is not None:
        fr, fc = game.food
        color = config.COLOR_FOOD_DEAD if dead else config.COLOR_FOOD
        pygame.draw.rect(
            surface, color,
            (ox + fc * cell + 1, oy + fr * cell + 1, cell - 2, cell - 2),
        )

    # Gövde — baş farklı renkte, yönünü görebilesin diye.
    for i, (r, c) in enumerate(game.body):
        if dead:
            color = config.COLOR_SNAKE_DEAD
        else:
            color = config.COLOR_HEAD if i == 0 else config.COLOR_BODY
        pygame.draw.rect(
            surface, color,
            (ox + c * cell + 1, oy + r * cell + 1, cell - 2, cell - 2),
        )

    # Çerçeve: lider farklı renkte.
    border = config.COLOR_LEADER if (is_leader and not dead) else config.COLOR_BORDER
    width = 3 if (is_leader and not dead) else 1
    pygame.draw.rect(surface, border, rect, width)

    if show_header:
        _draw_header(surface, game, rect, dead)


def _draw_header(surface, game, rect, dead):
    """Panelin üstüne kimlik, skor, adım yazar."""
    font = _get_font(max(11, rect.height // 22))

    label = getattr(game, "label", "")
    text = f"{label}  skor:{game.score}  adım:{game.steps}"
    if dead:
        text += f"  [{game.result.value}]"

    color = config.COLOR_TEXT_DEAD if dead else config.COLOR_TEXT
    surface.blit(font.render(text, True, color), (rect.x + 4, rect.y + 2))


# --------------------------------------------------------------------
# Izgarayı çizme
# --------------------------------------------------------------------

def draw_grid(surface, games, rects, leader_index=None):
    """Birden çok oyunu ızgarada çizer. games ve rects aynı sırada olmalı."""
    surface.fill(config.COLOR_BG)

    for i, (game, rect) in enumerate(zip(games, rects)):
        draw_game(surface, game, rect, is_leader=(i == leader_index))


def find_leader(games):
    """
    En yüksek skorlu CANLI oyunun indeksi.

    Not: bu sadece görsel bir işaret. Gerçek seçilim skora değil
    fitness'a göre yapılacak ve ikisi aynı şey değil.
    """
    best_i, best_score = None, -1
    for i, g in enumerate(games):
        if g.is_alive and g.score > best_score:
            best_i, best_score = i, g.score
    return best_i


# --------------------------------------------------------------------
# Yazı tipi önbelleği
# --------------------------------------------------------------------

_FONT_CACHE = {}


def _get_font(size):
    """
    Font'u her karede yeniden yaratmak pahalıdır — 9 panel x 60 fps
    demek saniyede 540 font nesnesi demek. Bir kere yarat, sakla.
    """
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.SysFont("menlo", size)
    return _FONT_CACHE[size]