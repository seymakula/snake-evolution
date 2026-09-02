"""
core/game.py — Yılan oyununun saf mantığı.

Bu dosyada pygame YOKTUR ve olmamalıdır.
Test: `import pygame` satırını sisteminden silsen bile bu dosya çalışmalı.
"""

import random

import config
from core import constants as C


class Game:
    """Tek bir yılan oyunu. Ekranı, ajanı, evrimi bilmez."""

    def __init__(
        self,
        rows=config.BOARD_ROWS,
        cols=config.BOARD_COLS,
        seed=None,
        obstacles=None,
    ):
        self.rows = rows
        self.cols = cols

        # Engeller: tahtanin ortasinda duran sabit bloklar.
        # Varsayilan BOS kume — yani engelsiz oyun, "engel listesi bos olan
        # oyun"dur. Bu sayede mevcut her sey aynen calismaya devam eder.
        # Kume, cunku her adimda "bu hucre engelde mi" diye soracagiz.
        self.obstacles = set(obstacles) if obstacles else set()

        # Her oyunun KENDİ rastgele üreteci. Global random kullanılmaz,
        # yoksa 9 oyun birbirinin rastgeleliğini etkiler.
        self.rng = random.Random(seed)

        # Açlık limiti tahta boyutuna bağlı olmalı: 3x3'te 800 adım anlamsız,
        # 20x20'de 20 adım çok kısa. Çarpanı config'den al.
        self.hunger_limit = self.rows * self.cols * config.HUNGER_FACTOR

        self.reset()

    # ------------------------------------------------------------------
    # Başlatma
    # ------------------------------------------------------------------

    def reset(self):
        """Oyunu başlangıç durumuna döndürür ve ilk durumu döndürür."""
        self.direction = self.rng.choice(C.DIRECTIONS)

        # Yılanı ortaya koy. Gövdeyi yönün TERSİNE doğru diz, yoksa
        # yılan ilk adımda kendi gövdesine girer.
        head = (self.rows // 2, self.cols // 2)
        self.body = [head]
        for i in range(1, config.SNAKE_LENGTH):
            r = head[0] - self.direction[0] * i
            c = head[1] - self.direction[1] * i
            if not self._in_bounds((r, c)) or (r, c) in self.obstacles:
                break
            self.body.append((r, c))

        # Liste sırayı, küme hızı verir. İkisi HER ZAMAN senkron tutulur.
        self.body_set = set(self.body)

        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.result = C.GameResult.RUNNING

        self.food = None
        self._place_food()

        return self.get_state()

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    @property
    def head(self):
        return self.body[0]

    @property
    def is_alive(self):
        return self.result == C.GameResult.RUNNING

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, pos):
        return not self._in_bounds(pos)

    def is_body(self, pos):
        return pos in self.body_set

    def is_obstacle(self, pos):
        return pos in self.obstacles

    def is_blocked(self, pos):
        """Bu hucre gecilemez mi? Duvar, engel ya da govde."""
        return self.is_wall(pos) or self.is_obstacle(pos) or self.is_body(pos)

    def is_danger(self, pos):
        """
        Bu hücreye girmek ölüm mü? Duyu vektöründe kullanılacak.

        Engel de burada sayiliyor — yoksa yilan engeli GOREMEZ ve
        ortam degisti ama ajan farkinda degil durumuna duseriz.
        """
        return self.is_blocked(pos)

    def _flood_fill(self, start, limit=None):
        """
        start hucresinden erisilebilecek bos hucre sayisi.

        Neden duz sayim degil: koridora girip saga donebiliyorsan o alan
        da erisilebilir. Duz sayim bunu goremez, flood fill gorur.
        Cikmaz sokagi gercekten tespit eden sey bu.

        limit: erken durdurma. "50'den fazla bos var" ile "87 bos var"
        arasindaki fark ajan icin onemsiz, ama hesap maliyeti buyuk.
        """
        if self.is_blocked(start):
            return 0

        if limit is None:
            limit = self.rows * self.cols

        gorulen = {start}
        kuyruk = [start]
        sayac = 0

        while kuyruk:
            r, c = kuyruk.pop()
            sayac += 1
            if sayac >= limit:
                break

            for dr, dc in C.DIRECTIONS:
                komsu = (r + dr, c + dc)
                # Engeller de dolu sayiliyor — yoksa alan sayaci
                # engelin arkasini da "erisilebilir" sanir.
                if komsu not in gorulen and not self.is_blocked(komsu):
                    gorulen.add(komsu)
                    kuyruk.append(komsu)

        return sayac

    def _place_food(self):
        """Boş bir hücreye yem koyar. Boş hücre yoksa oyun KAZANILMIŞTIR."""
        empty = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            # Engeller de haric — yoksa yem ulasilamaz bir yere duser.
            if (r, c) not in self.body_set and (r, c) not in self.obstacles
        ]

        if not empty:
            # Tahta tamamen doldu. Zaferin tanımı budur —
            # ayrı bir "kazandın mı" kontrolüne gerek yok.
            self.food = None
            self.result = C.GameResult.WON
            return

        self.food = self.rng.choice(empty)

    def _apply_action(self, action):
        """Göreli aksiyonu yeni yön vektörüne çevirir."""
        if action == C.GO_FORWARD:
            return self.direction

        idx = C.DIRECTION_INDEX[self.direction]
        if action == C.TURN_RIGHT:
            idx = (idx + 1) % len(C.DIRECTIONS)
        elif action == C.TURN_LEFT:
            idx = (idx - 1) % len(C.DIRECTIONS)
        else:
            raise ValueError(f"Geçersiz aksiyon: {action}")

        return C.DIRECTIONS[idx]

    # ------------------------------------------------------------------
    # Ana döngü adımı
    # ------------------------------------------------------------------

    def step(self, action):
        """
        Bir adım oynar.

        Dönüş: (state, reward, done, info)
        Bu dörtlü pekiştirmeli öğrenmenin standart sözleşmesidir.
        """
        # 1) Bitmiş oyuna adım attırma.
        if not self.is_alive:
            return self.get_state(), 0.0, True, self._info()

        # 2) Yönü güncelle.
        self.direction = self._apply_action(action)

        # 3) Yeni baş konumu = mevcut baş + yön.
        new_head = (
            self.head[0] + self.direction[0],
            self.head[1] + self.direction[1],
        )

        self.steps += 1
        self.steps_since_food += 1

        # 4) Duvara ya da ENGELE çarptı mı?
        # Ikisi de WALL sayiliyor: ajan acisindan ayni sey — gecilemez
        # sabit bir yuzey. Ayri etiket isteseydin constants'a eklerdin.
        if self.is_wall(new_head) or self.is_obstacle(new_head):
            self.result = C.GameResult.WALL
            return self.get_state(), config.DEATH_PENALTY, True, self._info()

        # 5) Kendine çarptı mı?
        # NOT: Kuyruğun bulunduğu hücreye girmek aslında ölüm DEĞİLDİR,
        # çünkü biz girerken kuyruk zaten çekiliyor. Denendi, geri alindi —
        # sebep icin README'ye bak.
        if self.is_body(new_head):
            self.result = C.GameResult.SELF
            return self.get_state(), config.DEATH_PENALTY, True, self._info()

        # 6) Başı ekle — listeye VE kümeye. Biri unutulursa hata
        # aylarca gizli kalır.
        self.body.insert(0, new_head)
        self.body_set.add(new_head)

        # 7) Yem yendi mi?
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward = config.FOOD_REWARD
            self._place_food()  # burada WON olabilir

            if self.result == C.GameResult.WON:
                return self.get_state(), reward, True, self._info()
        else:
            tail = self.body.pop()
            self.body_set.discard(tail)
            reward = config.STEP_REWARD

        # 8) Açlıktan öldü mü?
        if self.steps_since_food >= self.hunger_limit:
            self.result = C.GameResult.STARVED
            return self.get_state(), config.DEATH_PENALTY, True, self._info()

        # 9) Devam.
        return self.get_state(), reward, False, self._info()

    # ------------------------------------------------------------------
    # Ajanın gördüğü şey — 18 sayı
    # ------------------------------------------------------------------

    def get_state(self):
        """
        3 tehlike + 3 bos alan + 4 yon + 4 yem + 4 kuyruk = 18.

        Ham koordinat yerine bu sinyaller veriliyor: agin ogrenmesi
        gereken sey azaliyor ve ogrenilen kural tahta boyutundan
        bagimsiz kaliyor.
        """
        if self.food is None:
            return [0] * 18

        idx = C.DIRECTION_INDEX[self.direction]
        yon_direct = self.direction
        yon_right = C.DIRECTIONS[(idx + 1) % len(C.DIRECTIONS)]
        yon_left = C.DIRECTIONS[(idx - 1) % len(C.DIRECTIONS)]

        hr, hc = self.head
        direct_cell = (hr + yon_direct[0], hc + yon_direct[1])
        right_cell = (hr + yon_right[0], hc + yon_right[1])
        left_cell = (hr + yon_left[0], hc + yon_left[1])

        danger_direct = int(self.is_danger(direct_cell))
        danger_right = int(self.is_danger(right_cell))
        danger_left = int(self.is_danger(left_cell))

        # Bos alan sayaci: o yone gidersen ne kadar yer var?
        # Tehlike sensoru "dolu mu" der (ikili); bu "ne kadar yer var" der.
        # Cikmaz sokagi ancak bu gorebilir. Normalize ediliyor — ham sayi
        # 87 olurken diger duyular 0-1 arasinda kalsa ag icin dengesiz olurdu.
        toplam = self.rows * self.cols
        alan_direct = self._flood_fill(direct_cell) / toplam
        alan_right = self._flood_fill(right_cell) / toplam
        alan_left = self._flood_fill(left_cell) / toplam

        which_up = int(self.direction == C.UP)
        which_down = int(self.direction == C.DOWN)
        which_left = int(self.direction == C.LEFT)
        which_right = int(self.direction == C.RIGHT)

        fr, fc = self.food
        where_food_up = int(fr < hr)
        where_food_down = int(fr > hr)
        where_food_left = int(fc < hc)
        where_food_right = int(fc > hc)

        # Kuyruk yonu: kuyruga dogru gitmek genelde guvenlidir cunku
        # kuyruk cekiliyor. Bos alan "ne kadar yer var" der,
        # bu "nereden cikabilirim" der.
        tr, tc = self.body[-1]
        tail_up = int(tr < hr)
        tail_down = int(tr > hr)
        tail_left = int(tc < hc)
        tail_right = int(tc > hc)

        return [
            danger_direct, danger_left, danger_right,
            alan_direct, alan_left, alan_right,
            which_up, which_down, which_left, which_right,
            where_food_up, where_food_down, where_food_left, where_food_right,
            tail_up, tail_down, tail_left, tail_right,
        ]

    def _info(self):
        """Sonuç ekranı, fitness ve hata ayıklama için ekstra bilgi."""
        return {
            "score": self.score,
            "steps": self.steps,
            "steps_since_food": self.steps_since_food,
            "result": self.result,
            "length": len(self.body),
        }