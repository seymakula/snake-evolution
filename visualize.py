"""
visualize.py — Egitilmis sinir aginin katman diyagrami.

Standart NN semasindan farki: baglantilarin kalinligi ve rengi
GERCEK agirliklardan geliyor. Yani hem mimariyi hem ogrenilmis
beyni ayni gorselde gosteriyor.

Proje kokunden: python visualize.py
"""

import matplotlib.pyplot as plt
import numpy as np

import config
from agents.neural_agent import NeuralNetwork
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

# Baglantilarin yuzde kaci cizilsin. 168 baglantinin hepsini cizersen
# gorsel okunmaz hale gelir. Sadece EN GUCLU olanlari cizince
# "ag neye guveniyor" sorusunun cevabi ortaya cikar.
TOP_PERCENT = 25


def katman_konumlari(n, x, y_min=0.0, y_max=1.0):
    """
    Bir katmandaki n noronun (x, y) konumlarini uretir.

    Katmanlar farkli sayida noron iceriyor (11, 12, 3). Hepsini AYNI
    dikey araliga yayiyoruz ki hizali gorunsunler.
    """
    if n == 1:
        ys = np.array([(y_min + y_max) / 2])
    else:
        ys = np.linspace(y_max, y_min, n)
    return np.column_stack([np.full(n, x), ys])


def baglantilari_ciz(ax, sol, sag, W, esik):
    """
    Iki katman arasindaki agirliklari cizgi olarak cizer.

    linewidth  -> agirligin buyuklugu
    renk       -> pozitifse mavi, negatifse kirmizi
    alpha      -> zayif baglantilar soluk kalsin
    """
    max_w = np.abs(W).max()

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w = W[i, j]
            if abs(w) < esik:
                continue

            oran = abs(w) / max_w
            ax.plot(
                [sol[i, 0], sag[j, 0]],
                [sol[i, 1], sag[j, 1]],
                color="#2b6cb0" if w > 0 else "#c53030",
                linewidth=0.3 + 2.7 * oran,
                alpha=0.15 + 0.55 * oran,
                zorder=1,
            )


def noronlari_ciz(ax, konumlar, renk, boyut=420):
    ax.scatter(
        konumlar[:, 0],
        konumlar[:, 1],
        s=boyut,
        c=renk,
        edgecolors="#1a202c",
        linewidths=1.2,
        zorder=3,
    )


def main():
    # --- Egitilmis agi yukle ---
    genome = np.load("models/best.npy")
    net = NeuralNetwork(config.STATE_SIZE, config.HIDDEN_SIZE, C.ACTION_NUM)
    net.from_vector(genome)  # BU SATIR OLMAZSA rastgele agi cizersin!

    # --- Esikler: en guclu %25 ---
    esik1 = np.percentile(np.abs(net.w1), 100 - TOP_PERCENT)
    esik2 = np.percentile(np.abs(net.w2), 100 - TOP_PERCENT)

    # --- Noron konumlari ---
    girdi = katman_konumlari(config.STATE_SIZE, 0.0)
    gizli = katman_konumlari(config.HIDDEN_SIZE, 1.0)
    cikti = katman_konumlari(C.ACTION_NUM, 2.0, 0.30, 0.70)

    fig, ax = plt.subplots(figsize=(13, 8))

    baglantilari_ciz(ax, girdi, gizli, net.w1, esik1)
    baglantilari_ciz(ax, gizli, cikti, net.w2, esik2)

    noronlari_ciz(ax, girdi, "#90cdf4")
    noronlari_ciz(ax, gizli, "#d6bcfa")
    noronlari_ciz(ax, cikti, "#fbd38d", boyut=620)

    # --- Etiketler ---
    for (x, y), isim in zip(girdi, DUYULAR):
        ax.text(x - 0.06, y, isim, ha="right", va="center", fontsize=10)

    for k, (x, y) in enumerate(gizli):
        ax.text(x, y, str(k), ha="center", va="center", fontsize=8, zorder=4)

    for (x, y), isim in zip(cikti, AKSIYONLAR):
        ax.text(x + 0.06, y, isim, ha="left", va="center", fontsize=11, fontweight="bold")

    # --- Katman basliklari ---
    ax.text(
        0.0, 1.09, f"GIRDI\n{config.STATE_SIZE} duyu", ha="center", fontsize=11, fontweight="bold"
    )
    ax.text(
        1.0, 1.09, f"GIZLI\n{config.HIDDEN_SIZE} noron", ha="center", fontsize=11, fontweight="bold"
    )
    ax.text(
        2.0, 1.09, f"CIKTI\n{C.ACTION_NUM} aksiyon", ha="center", fontsize=11, fontweight="bold"
    )

    ax.set_title(
        f"Egitilmis yilan beyni — en guclu %{TOP_PERCENT} baglanti\n"
        f"mavi = pozitif agirlik, kirmizi = negatif, kalinlik = buyukluk",
        fontsize=12,
        pad=28,
    )

    ax.set_xlim(-0.55, 2.5)
    ax.set_ylim(-0.10, 1.22)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("nn_map.png", dpi=150, bbox_inches="tight")
    print("Kaydedildi: nn_map.png")
    plt.show()

    # --- Rakamla ozet: hangi duyu daha etkili? ---
    etki = np.abs(net.w1).sum(axis=1)
    sira = np.argsort(etki)[::-1]

    print("\nDuyularin toplam etkisi (buyukten kucuge):")
    for i in sira:
        cubuk = "#" * int(etki[i] * 2)
        print(f"  {DUYULAR[i]:16s} {etki[i]:6.2f}  {cubuk}")


if __name__ == "__main__":
    main()
