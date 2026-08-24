"""
plot_history.py — Ogrenme egrisi.

train.py'nin kaydettigi models/history.npy dosyasini okuyup iki panel cizer:
  sol  : fitness (secilim olcutu, soyut)
  sag  : skor (kac yem yendi, somut)

Bakilacak sey: EGRININ SEKLI. Genetik algoritmalarda tipik desen
uzun bir PLATO, sonra ani SICRAMA'dir. Plato bosa gecmez — populasyonda
ise yarar agirlik parcalari birikir; bir caprazlama dogru parcalari
birlestirdiginde her sey degisir.

Proje kokunden: python plot_history.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

YOL = "models/history.npy"


def main():
    if not os.path.exists(YOL):
        print(f"{YOL} bulunamadi.")
        print("Once train.py'a history kaydetme satirini ekleyip calistir.")
        return

    h = np.load(YOL)
    avg_fit, best_fit, avg_score, best_score = h[:, 0], h[:, 1], h[:, 2], h[:, 3]
    nesil = np.arange(len(h))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Sol: fitness ---
    ax1.plot(nesil, best_fit, color="#2b6cb0", linewidth=1.6, label="en iyi birey")
    ax1.plot(nesil, avg_fit, color="#c53030", linewidth=2.0, label="populasyon ortalamasi")
    ax1.fill_between(nesil, avg_fit, alpha=0.12, color="#c53030")
    ax1.set_xlabel("nesil")
    ax1.set_ylabel("fitness")
    ax1.set_title("Fitness — secilim olcutu")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # --- Sag: skor (yenen yem) ---
    ax2.plot(nesil, best_score, color="#2f855a", linewidth=1.6, label="en iyi birey")
    ax2.plot(nesil, avg_score, color="#dd6b20", linewidth=2.0, label="populasyon ortalamasi")
    ax2.fill_between(nesil, avg_score, alpha=0.12, color="#dd6b20")
    ax2.set_xlabel("nesil")
    ax2.set_ylabel("yenen yem")
    ax2.set_title("Skor — somut basari")
    ax2.legend()
    ax2.grid(alpha=0.25)

    # --- Sicrama noktasini isaretle ---
    # Ortalamanin baslangic seviyesinin 3 katini ilk astigi nesil.
    esik = avg_fit[:5].mean() * 3
    ustunde = np.where(avg_fit > esik)[0]
    if len(ustunde):
        kirilma = ustunde[0]
        for ax in (ax1, ax2):
            ax.axvline(kirilma, color="#718096", linestyle="--", linewidth=1)
        ax1.text(kirilma + 2, ax1.get_ylim()[1] * 0.9,
                 f"sicrama\n~nesil {kirilma}", fontsize=9, color="#4a5568")
        print(f"Sicrama noktasi: nesil {kirilma}")

    fig.suptitle(
        f"Ogrenme egrisi — {len(h)} nesil   "
        f"(ort. fitness {avg_fit[0]:.2f} -> {avg_fit[-1]:.2f})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("ogrenme_egrisi.png", dpi=150, bbox_inches="tight")
    print("Kaydedildi: ogrenme_egrisi.png")
    plt.show()

    # --- Rakamla ozet ---
    print(f"\nNesil   0: ort fitness {avg_fit[0]:7.2f}   ort skor {avg_score[0]:5.2f}")
    print(f"Nesil {len(h)//2:>3}: ort fitness {avg_fit[len(h)//2]:7.2f}   ort skor {avg_score[len(h)//2]:5.2f}")
    print(f"Nesil {len(h)-1:>3}: ort fitness {avg_fit[-1]:7.2f}   ort skor {avg_score[-1]:5.2f}")
    print(f"\nEn yuksek skor: {int(best_score.max())} (nesil {int(best_score.argmax())})")


if __name__ == "__main__":
    main()
