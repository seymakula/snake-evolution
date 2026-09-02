"""
train.py — Ekransiz egitim + checkpoint.

Egitim kaldigi yerden devam eder. Checkpoint'te sadece en iyi genom
degil, POPULASYONUN TAMAMI saklanir — yoksa evrim yeni bir populasyonla
bastan baslar ve kaldigi yerden devam etmis olmaz.

Checkpoint yolu: models/checkpoint_train.npz
evolve_live.py AYRI bir dosyaya yazar (models/checkpoint.npz), boylece
ekranda birkac nesil izlemek buradaki 300 nesillik egitimi ezmez.

Kullanim:
    python train.py            # checkpoint varsa devam eder
    python train.py --fresh    # checkpoint'i yok sayar, sifirdan baslar

Ctrl+C ile durdurursan son durum kaydedilir.
"""

import ast
import os
import sys
import csv
import numpy as np

import config
from core import constants as C
from evolution.population import evaluate_population
from evolution.genetic import next_generation

CHECKPOINT_TRAIN = "models/checkpoint_train.npz"
CHECKPOINT_LIVE = "models/checkpoint.npz"


# ----------------------------------------------------------------------
# Genom uretimi
# ----------------------------------------------------------------------

def genome_size():
    return (
        config.STATE_SIZE * config.HIDDEN_SIZE
        + config.HIDDEN_SIZE
        + config.HIDDEN_SIZE * C.ACTION_NUM
        + C.ACTION_NUM
    )


def random_genomes(n, seed=0):
    """n tane rastgele genom uret. Nesil 0'in ham maddesi."""
    rng = np.random.default_rng(seed)
    size = genome_size()
    return [rng.normal(0, 1, size) for _ in range(n)]


# ----------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------
def save_csv(history, yol="models/history.csv"):
    """
    Egitim gecmisini CSV olarak yazar.

    .npy sadece Python'dan okunur; CSV'yi Excel'de acabilir, baska bir
    dile tasiyabilir, kendi grafigini cizebilirsin.
    """
    os.makedirs("models", exist_ok=True)

    with open(yol, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nesil", "ort_fitness", "en_iyi_fitness",
                    "ort_skor", "en_iyi_skor"])
        for i, satir in enumerate(history):
            w.writerow([i] + [round(float(v), 4) for v in satir])
def save_checkpoint(genomes, gen, history, rng, best_fitness, yol=CHECKPOINT_TRAIN):
    """
    Devam edebilmek icin gereken HER SEYI kaydeder.

    rng durumu neden onemli: kaydetmezsen devam ettiginde ayni rastgele
    diziyi bastan uretir ve tekrarlanabilirlik bozulur.
    np.savez sozluk saklamaz, o yuzden str() ile metne cevriliyor.

    yol parametresi: train.py ve evolve_live.py ayri dosyalara yazsin diye.
    """
    os.makedirs("models", exist_ok=True)

    np.savez(
        yol,
        genomes=np.array(genomes),
        gen=gen,
        history=np.array(history) if history else np.zeros((0, 4)),
        rng_state=str(rng.bit_generator.state),
        best_fitness=best_fitness,
        # Mimari bilgisi: yuklerken uyusma kontrolu icin.
        hidden_size=config.HIDDEN_SIZE,
        state_size=config.STATE_SIZE,
    )


def load_checkpoint(yol=CHECKPOINT_TRAIN):
    """Checkpoint varsa sozluk doner, yoksa None."""
    if not os.path.exists(yol):
        return None

    d = np.load(yol, allow_pickle=False)

    # HIDDEN_SIZE ya da STATE_SIZE degistirilip eski checkpoint'ten devam
    # edilirse genom boyu tutmaz ve from_vector anlasilmaz bir hata verir.
    # Bu kontrol onu onceden yakalar.
    if int(d["hidden_size"]) != config.HIDDEN_SIZE or int(d["state_size"]) != config.STATE_SIZE:
        print("Checkpoint farkli mimariye ait — sifirdan baslaniyor.")
        print(
            f"  kayitli: {int(d['state_size'])}->{int(d['hidden_size'])}, "
            f"simdiki: {config.STATE_SIZE}->{config.HIDDEN_SIZE}"
        )
        return None

    # literal_eval kullaniliyor, eval degil — eval herhangi bir kodu
    # calistirir, guvenli degil.
    state = ast.literal_eval(str(d["rng_state"]))
    rng = np.random.default_rng()
    rng.bit_generator.state = state

    return {
        "genomes": [g for g in d["genomes"]],
        "gen": int(d["gen"]),
        "history": [list(row) for row in d["history"]],
        "rng": rng,
        "best_fitness": float(d["best_fitness"]),
    }


# ----------------------------------------------------------------------
# Egitim
# ----------------------------------------------------------------------

def main():
    os.makedirs("models", exist_ok=True)

    ck = None
    if "--fresh" not in sys.argv:
        ck = load_checkpoint()

    if ck is not None:
        genomes = ck["genomes"]
        baslangic = ck["gen"] + 1
        history = ck["history"]
        rng = ck["rng"]
        best_fitness = ck["best_fitness"]
        print(f"Checkpoint bulundu — nesil {baslangic}'ten devam ediliyor.")
        print(f"(su ana kadarki en iyi fitness: {best_fitness:.3f})\n")
    else:
        rng = np.random.default_rng(config.TRAIN_SEED)
        genomes = random_genomes(config.POPULATION_SIZE, seed=config.TRAIN_SEED)
        baslangic = 0
        history = []
        best_fitness = float("-inf")
        print("Sifirdan baslaniyor.\n")

    if baslangic >= config.GENERATION_LIMIT:
        print(
            f"Zaten {baslangic} nesil tamamlanmis. "
            f"config.GENERATION_LIMIT'i artir ya da --fresh kullan."
        )
        return

    gen = baslangic

    try:
        for gen in range(baslangic, config.GENERATION_LIMIT):
            results, stats = evaluate_population(genomes, seed=gen)

            print(
                f"Nesil {gen:>3} | "
                f"fitness ort {stats['avg_fitness']:>6.3f} "
                f"en iyi {stats['best_fitness']:>7.3f} | "
                f"skor ort {stats['avg_score']:>5.2f} "
                f"en iyi {stats['best_score']:>2} | "
                f"adim ort {stats['avg_steps']:>5.1f}"
            )

            if gen % 10 == 0:
                deaths = "  ".join(
                    f"{name}:{count}" for name, count in stats["deaths"].most_common()
                )
                print(f"          olumler: {deaths}")

            if stats["best_fitness"] > best_fitness:
                best_fitness = stats["best_fitness"]
                np.save("models/best.npy", results[0][2])


            history.append(
                [
                    stats["avg_fitness"],
                    stats["best_fitness"],
                    stats["avg_score"],
                    stats["best_score"],
                ]
            )

            # Her nesilde diske yazmak gereksiz yuk; 10'da bir yeterli.
            # Ctrl+C durumu asagida ayrica yakalaniyor.
            if gen % 10 == 0:
                save_checkpoint(genomes, gen, history, rng, best_fitness)

            genomes = next_generation(results, rng)

    except KeyboardInterrupt:
        print("\n\nDurduruldu — kaydediliyor...")
        save_checkpoint(genomes, gen, history, rng, best_fitness)
        np.save("models/history.npy", np.array(history))
        save_csv(history)
        print(f"Nesil {gen}'e kadar kaydedildi. Tekrar 'python train.py' ile devam edebilirsin.")
        return

    save_checkpoint(genomes, config.GENERATION_LIMIT - 1, history, rng, best_fitness)
    np.save("models/history.npy", np.array(history))
    save_csv(history)

    print()
    print("=" * 62)
    print(f"Egitim bitti — nesil {baslangic}..{config.GENERATION_LIMIT - 1}")
    print(f"En iyi fitness    : {best_fitness:.3f}")
    print(f"Ilk nesil ortalama: {history[0][0]:.3f}")
    print(f"Son nesil ortalama: {history[-1][0]:.3f}")
    print(f"En iyi skor       : {int(max(h[3] for h in history))}")
    print("Kaydedildi        : models/best.npy, models/history.npy")
    print(f"Checkpoint        : {CHECKPOINT_TRAIN}")
    print("=" * 62)


if __name__ == "__main__":
    main()