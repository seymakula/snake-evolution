"""
test_game.py — Oyun motorunun doğruluk testleri.

Proje kökünden çalıştır:  python test_game.py

DİKKAT: Bu dosya pygame import ETMEZ. Motor da etmez.
Testler ekransız çalışır — eğitim de tam olarak böyle çalışacak.
"""

import random

from core.game import Game
from core import constants as C


def test_no_crash(n_games=500):
    """Rastgele aksiyonlarla 500 oyun. Hiçbiri çökmemeli."""
    results = {}

    for i in range(n_games):
       
        game = Game(seed=i)
        rng = random.Random(i)

        while game.is_alive:
            state = game.get_state()
            assert len(state) == 18, "duyu vektörü 11 eleman olmalı"
            #assert all(v in (0, 1) for v in state), "değerler 0 veya 1 olmalı"
            assert sum(state[6:10]) == 1, "tam bir yön aktif olmalı"

            action = rng.choice([C.GO_FORWARD, C.TURN_RIGHT, C.TURN_LEFT])
            game.step(action)
            if not game.is_alive:
                break

        # Oyun bittiyse RUNNING dışında bir sonuçla bitmiş olmalı.
        assert game.result != C.GameResult.RUNNING, f"Oyun {i} bitmedi"
        results[game.result] = results.get(game.result, 0) + 1

    print(f"[OK] {n_games} oyun çöktürmeden bitti.")
    for result, count in results.items():
        print(f"     {result.value:10s} {count}")
    return results


def test_different_seeds():
    """Farklı tohumlu oyunlar bağımsız olmalı — yemler farklı yerlerde."""
    foods = [Game(seed=s).food for s in range(9)]

    assert len(set(foods)) > 1, (
        "9 oyunun ilk yemi de aynı yerde! Oyunlar bağımsız değil — "
        "muhtemelen global random kullanılıyor."
    )

    print(f"[OK] 9 farklı tohum, {len(set(foods))} farklı yem konumu.")


def test_same_seed_reproducible():
    """Aynı tohum + aynı aksiyonlar = birebir aynı sonuç."""

    def play(seed):
        game = Game(seed=seed)
        rng = random.Random(999)
        while game.is_alive:
            game.step(rng.choice([C.GO_FORWARD, C.TURN_RIGHT, C.TURN_LEFT]))
        return game.score, game.steps, game.result

    a = play(42)
    b = play(42)

    assert a == b, f"Aynı tohum farklı sonuç verdi: {a} != {b}"
    print(f"[OK] Tekrarlanabilir: {a}")


def test_body_set_sync():
    """body listesi ile body_set kümesi her adımda senkron kalmalı."""
    game = Game(seed=7)
    rng = random.Random(7)

    while game.is_alive:
        game.step(rng.choice([C.GO_FORWARD, C.TURN_RIGHT, C.TURN_LEFT]))
        assert len(game.body) == len(game.body_set), (
            f"Senkron bozuldu: liste {len(game.body)}, küme {len(game.body_set)}. "
            "Bir yerde insert/add veya pop/discard çifti eksik."
        )
        assert set(game.body) == game.body_set, "İçerikler uyuşmuyor."

    print(f"[OK] body ve body_set {game.steps} adım boyunca senkron kaldı.")


def test_win_on_tiny_board():
    """
    3x3 tahtada kazanmak mümkün mü? Rastgele oynayarak dene.
    Kazanan çıkarsa WON etiketi ve zafer ekranı doğru çalışıyor demektir.
    """
    wins = 0
    for i in range(3000):
        game = Game(rows=3, cols=3, seed=i)
        rng = random.Random(i)
        while game.is_alive:
            game.step(rng.choice([C.GO_FORWARD, C.TURN_RIGHT, C.TURN_LEFT]))
        if game.result == C.GameResult.WON:
            wins += 1

    print(f"[OK] 3x3 tahtada 3000 rastgele oyunda {wins} zafer.")
    if wins == 0:
        print("     UYARI: hiç kazanan yok. Rastgele için normal olabilir,")
        print("     ama _place_food içindeki WON kontrolünü gözden geçir.")


def test_no_pygame():
    """Motor pygame'e bağımlı olmamalı."""
    import sys

    assert "pygame" not in sys.modules, (
        "pygame yüklenmiş! core/ içinde bir yerde import ediliyor. " "Motor ekransız çalışabilmeli."
    )
    print("[OK] Motor pygame'siz çalışıyor.")


if __name__ == "__main__":
    test_no_pygame()
    test_different_seeds()
    test_same_seed_reproducible()
    test_body_set_sync()
    test_no_crash()
    test_win_on_tiny_board()
    print("\nHepsi geçti.")
