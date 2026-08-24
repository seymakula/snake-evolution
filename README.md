# Snake Evolution

Genetik algoritma ile kendi kendine yılan oynamayı öğrenen sinir ağı ajanı. Backpropagation yok — ağırlıklar nesiller boyunca seçilim, çaprazlama ve mutasyonla evrimleşiyor.

![Öğrenme eğrisi](ogrenme_egrisi.png)

300 nesilde ortalama fitness 0.63 → 46.33. En iyi birey 68 yem (100 hücrelik tahtada).

---

## Kurulum

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

| Komut | Ne yapar |
|---|---|
| `python train.py` | Ekransız eğitim. `models/best.npy` ve `models/history.npy` üretir. |
| `python evolve_live.py` | Evrimi ekranda izle — nesiller ilerledikçe yılanlar iyileşir. |
| `python main.py` | Kaydedilmiş en iyi beyni izle. |
| `python plot_history.py` | Öğrenme eğrisini çiz. |
| `python visualize.py` | Ağırlık haritasını çiz. |
| `python test_game.py` | Oyun motorunun doğruluk testleri. |

Tuşlar: `P` duraklat · `SPACE` hız (1x/2x/4x) · `I` çık · `N` nesli atla

Tüm ayarlar `config.py` içinde.

---

## Mimari

Üç ayrım projenin tamamını belirledi:

**Mantık ≠ görüntü.** `core/game.py` içinde tek bir pygame satırı yok. Eğitim tamamen ekransız koşuyor — 100 birey × 300 nesil dakikalar alıyor, ekrana çizilseydi saatler sürerdi.

**Ortam ≠ ajan.** `agents/base.py` bir sözleşme tanımlıyor: durum ver, aksiyon al. `RandomAgent` ve `NeuralAgent` aynı yerden takılıyor; sinir ağı eklenirken motorda tek satır değişmedi.

**Kod ≠ ayar.** Hiperparametrelerin tamamı `config.py`'da. Deney yapmak tek satır değiştirmek.

```
core/       oyun motoru + sabitler   (pygame YOK)
agents/     ajan sözleşmesi, rastgele ajan, sinir ağı
evolution/  fitness, popülasyon, genetik operatörler
render/     çizim
```

---

## Yöntem

### Duyu vektörü (11 sayı)

| Grup | İçerik |
|---|---|
| Tehlike (3) | önde / sağda / solda ölüm var mı |
| Yön (4) | yılan hangi yöne bakıyor (one-hot) |
| Yem (4) | yem başa göre yukarıda / aşağıda / solda / sağda mı |

Ham koordinat (`baş=(7,3)`, `yem=(2,9)`) yerine bu sinyaller veriliyor. İki sebeple: ağın öğrenmesi gereken şey azalıyor, ve öğrenilen kural tahta boyutundan bağımsız kalıyor.

Yön için tek sayı (0-3) yerine dört ayrı kutu kullanılıyor — tek sayı verilse ağ "sağ (3), yukarıdan (0) üç kat fazla bir şey" gibi anlamsız bir sıra ilişkisi kurardı.

### Ağ

`11 → 12 → 3`, gizli katmanda `tanh`. Toplam **183 parametre** (132 + 12 + 36 + 3).

`tanh` olmadan iki matris çarpımı matematiksel olarak tek matrise çöker ve gizli katman işlevsiz kalır. Gizli katman "önümde tehlike VAR **ve** sağımda da VAR ise sola dön" gibi bileşik kurallar için gerekli.

Çıktının en büyüğünün **indeksi** doğrudan aksiyon kodu (`GO_FORWARD=0`, `TURN_RIGHT=1`, `TURN_LEFT=2`), çeviri tablosu yok.

### Fitness

```
fitness = skor × 2.0 + adım × 0.005
```

Sadece skor kullanılsa seçilim imkânsız olurdu: ilk testte 100 yılanın 89'unun skoru 0 çıktı, hepsi eşitti. Adım terimi bu bireyleri birbirinden ayırıyor.

Katsayı kasıtlı olarak çok küçük. Şart: `maksimum_adım × katsayı < bir_yemin_değeri`. Aksi halde yılan yem yemek yerine sonsuza kadar daire çizmeyi öğrenir — ki bu gerçekten oldu, aşağıya bakın.

### Evrim (nesil başına)

1. 100 birey oynatılır, fitness'a göre sıralanır
2. En iyi 4 aynen taşınır (**elitizm**) — en iyi çözüm asla kaybedilmez
3. Kalan 96: turnuva seçilimi (k=5) → uniform crossover → mutasyon (oran 0.05, güç 0.2)

Turnuva havuzu tüm popülasyon, sadece elitler değil. Sıralamada 60. olan bir birey genel olarak kötü olsa da işe yarar bir parça taşıyabilir; çaprazlama o parçayı iyi bir genomla birleştirebilir.

Nesiller arasında taşınan **tek şey genom**. Gövde uzunluğu, skor, adım sayısı — hiçbiri taşınmaz. Her yılan sıfırdan doğar.

---

## Bulgular

### 1. Plato, sonra sıçrama

Öğrenme doğrusal değil. **45 nesil boyunca hiçbir yılan 2'den fazla yem yiyemedi.** Sonra:

- Nesil ~45: **tek bir birey** çözümü buldu (mavi çizgi yükselir)
- Nesil ~53: genler popülasyona yayıldı (kırmızı çizgi yükselir)

Plato boşa geçmiyor — popülasyonda işe yarar ağırlık parçaları birikiyor, bir çaprazlama doğru parçaları birleştirdiğinde çözülüyor. Nesil 30'da "çalışmıyor" deyip durdurulsaydı hiçbir şey görülmeyecekti.

En iyi birey ile ortalama arasındaki kalıcı boşluk (sonda 120'ye karşı 46) çeşitliliğin göstergesi. Kapansaydı popülasyon tek çözüme saplanır, evrim dururdu.

### 2. Ölüm sebebi dağılımı — en öğretici bulgu

| Nesil | wall | starved | self |
|---|---|---|---|
| 0 | 60 | 38 | 2 |
| 30 | 32 | **65** | 3 |
| 60 | 30 | 23 | 47 |
| 90 | 16 | 7 | **77** |

Üç ayrı öğrenme aşaması:

**Duvardan kaçmayı öğrendiler.** `wall` 60'tan 32'ye düştü.

**Sonra daire çizmeye başladılar.** Nesil 30'da `starved` 65 — yılanlar hayatta kalmayı çözdü ama yem yemedi. Bu, adım teriminin yarattığı **ödül sömürüsü** (reward hacking): ajan istenen şeyi değil, puan verilen şeyi yapıyor.

Kendiliğinden düzeldi çünkü katsayı doğru ayarlıydı: bir yem 2 puan, 200 adım daire yalnızca 1 puan. Daire çizenler bir süre öne geçti, yem yiyenler ortaya çıkınca elendiler.

**En sonda kendilerine sıkışmaya başladılar.** `self` 77 — ve bu bir **başarı göstergesi**: kendine çarpabilmek için uzun olman, uzun olmak için çok yem yemiş olman gerekiyor.

Tek bir fitness sayısı "iyi mi kötü mü" der; ölüm dağılımı "neden" der.

### 3. Hiperparametre deneyleri

**Gizli katman boyutu** (100 nesil):

| Boyut | Genom | Son nesil ort. | En iyi skor |
|---|---|---|---|
| 6 | 99 | 3.29 | 8 |
| **12** | **183** | **46.25** | **68** |
| 24 | 363 | 35.52 | 34 |

6'da ağ kapasitesi bileşik kurallara yetmiyor. 24'te genom iki katına çıktığı için genetik algoritma aynı nesil sayısında yakınsayamıyor.

**Mutasyon oranı** (100 nesil):

| Oran | Son nesil ort. | En iyi skor |
|---|---|---|
| 0.01 | **62.91** | 33 |
| **0.05** | 46.25 | **68** |
| 0.15 | 49.95 | 51 |

İki metrik ters yönde hareket ediyor. Düşük mutasyon popülasyonu homojenleştiriyor — ortalama yüksek ama tavan kırılmıyor. Yüksek mutasyon iyi çözümleri bozuyor.

Bu projede tek bir beyin kaydedilip gösterildiği için **tepe** optimize edildi → 0.05.

---

## Denenip geri alınan: çok oyunlu değerlendirme

**Hipotez:** Her genom tek oyunla değerlendiriliyor. Bir genom o özel yem dizilimine iyi denk geldiği için yüksek fitness alabilir — aşırı uyum riski.

**Deney:** Her genom 3 farklı tohumla oynatıldı, fitness ortalaması alındı (`train_multi.py`). Sonuçlar **eğitimde hiç görülmemiş** 30 tohumda ölçüldü (`genelleme_testi.py`).

| | Tek oyunlu (100 nesil) | Çok oyunlu (100 nesil) | Çok oyunlu (300 nesil) |
|---|---|---|---|
| ortalama skor | **42.63** | 21.20 | 31.77 |
| std sapma | 13.14 | **2.43** | 8.74 |

**Sonuç: hipotez doğru ama getirisi maliyetini karşılamadı.**

100 nesildeki düşük std sapma yanıltıcıydı — model henüz öğrenmediği için her tohumda benzer şekilde ortalama oynuyordu. Nesil sayısı üçe katlanınca ortalama toparlandı ama tek oyunluyu yakalayamadı; üstelik dokuz kat daha fazla hesap harcayarak.

Muhtemel sebep: 10×10 tahtada tek bir oyun yüzlerce adım sürüyor ve yılan onlarca farklı durumla karşılaşıyor — yani tek oyun zaten yeterli sinyal veriyor.

Tek oyunlu değerlendirme korundu. Kod karşılaştırma için repoda bırakıldı.

---

## Bilinen sınırlar

**Yılan sonunda kendi gövdesine sıkışıyor.** Bir hata değil, duyu vektörünün sınırı: yılan yalnızca bir adım ötesini görüyor, kendi gövdesinin şeklini görmüyor. Girdiği boşluğun çıkmaz sokak olup olmadığını bilemiyor.

Çözüm için duyu vektörü genişletilebilir — her yönde kaç boş hücre olduğu, ya da kuyruğun yönü. İkisi de vektörü büyütür ve öğrenmeyi yavaşlatır.

**Deneyler tek koşu.** Genetik algoritma rastgele bir süreç; yukarıdaki farkların gerçek mi gürültü mü olduğu belirsiz. Sağlam sonuç için her ayar 3 farklı `TRAIN_SEED` ile tekrarlanmalı.

**`evolve_live.py` ekranda 9 yılan gösteriyor ama popülasyon 100.** 9 bireyle elitizm popülasyonun yarısını kaplar, çeşitlilik ölür ve evrim yakınsamaz. Ekrandakiler o neslin en iyi 9'u.

---

## Ağırlık haritası

![NN haritası](nn_map.png)

Eğitilmiş ağın en güçlü %25 bağlantısı. Mavi pozitif, kırmızı negatif, kalınlık büyüklük.

`visualize.py` ayrıca her duyunun toplam etkisini (`|w1|` satır toplamları) rakamla listeliyor — yani yılanın karar verirken en çok neye baktığını.

---

## Kaynaklar
- Code Bullet (YouTube): Connect 4 / Jump King / Happy Wheels — arama, seyrek ödül ve neuroevolution yaklaşımlarının karşılaştırması