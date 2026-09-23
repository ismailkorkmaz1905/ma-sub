# Otomatik resmî altyazı akışı

**PC, RunPod, Colab oturumu veya ücretli çeviri API'si gerekmez.** Zamanlanmış
ChatGPT görevi resmî bölümü kontrol eder, Türkçe WebVTT hazırsa mevcut üretim
talimatlarıyla çevirir ve Drive'a kaydeder. Python indirme, kimlik/zaman kontrolü
ve SRT üretimini yapar. Çeviri ChatGPT'nin kendisindedir; cron içinden ChatGPT
aboneliğine API çağrısı yapıldığı iddia edilmez.

## Zamanlama

İlk hedef **15. bölüm**, başlangıç **25 Eylül 2026 cuma 06:00 Asia/Singapore**.
Tek görev saatlik kontrol eder; bölüm veya Türkçe dosya yoksa bekler. İş bittiğinde
yeniden çevrilmez. Kontrol penceresi dört günle sınırlıdır. Saat, hazır altyazının
teslim saati değildir; yayıncının Türkçe dosyayı yükleme saatine bağlıdır.

Bu zamanlayıcı ChatGPT Tasks'tır; GitHub Actions, ek sunucu veya ikinci cron yoktur.
Görev kapsamı bu bölüm içindir. Takip bölümleri ayrıca hedeflenir.

## Küçük komutlar

```bash
PYTHONPATH=src python -m mas.official_subtitles prepare --episode 15 --root work/ep15
PYTHONPATH=src python -m mas.official_subtitles pending --root work/ep15
PYTHONPATH=src python -m mas.official_subtitles finish --root work/ep15
```

`prepare` resmî tanıtım sayfasındaki gerçek bölüm bağlantısını bulur; URL'de
bölüm numarası tahmin ederek fragman seçmez. `WAITING_FOR_EPISODE` veya
`WAITING_FOR_TURKISH_CAPTIONS` normal bekleme durumlarıdır. Bölüm var ama Türkçe
altyazı yoksa ASR/ücretli servis başlatılmaz.

- Kaynak: yayıncının `.vtt` dosyası. Kelime zamanı uydurma, ASR veya forced alignment yok.
- Çeviri: değişmeyen üretim talimatları/sözlük; 60 satırlık tamamlanan parçalar kaydedilir.
- Zamanlar: kaynaktan aynen aktarılır. Uzun çeviri için süre uzatılmaz.
- Teslim: Endonezce/Türkçe SRT + açık sorun raporu; eksik parça başarı sayılmaz.
- Yeniden başlama: Drive'daki `ep15-official-work.zip` indirilir ve ilk eksik parçadan sürer.
- Gizlilik: mevcut özel Drive klasörü kullanılır; paylaşım izinleri değiştirilmez.

## Önemli kaynak sınırı

Show TV sürümü ile başka bir YouTube/Drive sürümü aynı olmayabilir. Çıktıda
`SHOWTV` yazmasının nedeni budur. Kaynak medya URL'si, yayıncı asset ID'si,
süre ve VTT SHA-256 kaydedilir. Başka kurguya doğrulanmadan senkron denmez.

Show TV oynatıcısı bu oturumda Singapur için bölge engeli gösterdi. Video erişim
engeli aşılmaz. Herkese açık Türkçe altyazı dosyası indirilebildi. Kullanıcının
Drive videosu tarayıcıda oynadı; süre eşleşmesi tek başına senkron kanıtı değildir.

`TRANSLATED_WITH_PUBLISHER_TIMES`, zamanların yayıncıdan korunduğu anlamındadır;
konuşmadan önce/sonra gerçek hata ölçümü veya tam bölüm dinleme onayı değildir.
Kalite sorunu varsa dosya `.draft.srt` olur. Görev bunu nihai kalite onayı diye sunmaz.

## Yapılan gerçek deneme (23 Eylül)

- 14. bölümün sayfası ve Türkçe VTT'si canlı indirildi: **2.980 cue**, **193.595 bayt**.
- VTT SHA-256: `719ae8728279db437f1063fd35fa154a59be6efca26e35bc2a94b16b5fe484ac`.
- 01:16.358-01:35.678 arasındaki 6 gerçek satır ChatGPT ile çevrildi ve SRT üretildi.
- 6/6 başlangıç-bitiş aralığı birebir korundu. Çeviriye eklenen zaman alanı reddedildi.
- Eksik bölüm çevirisiyle `finish` çalıştırıldı; başarı vermedi.
- 15. bölüm kontrolü canlı çalıştırıldı: `WAITING_FOR_EPISODE`.
- Örnek ZIP Drive'a yüklendi ve bağımsız indirmeyle bayt bayt karşılaştırıldı.
- Ses dinlenerek zamanlama testi **yapılamadı**. Drive indirme bağlantısı 256 MiB
  sınırında 4,95 GB dosyayı reddetti; tarayıcı indirme girişimi de zaman aşımına uğradı.

İnceleme örneği: `Muhtemel_Ask_14_gercek_altyazi_ornegi.zip`.
Üretim için hâlâ gerekli kanıt: erişilebilen aynı video kopyasıyla kısa sesli
karşılaştırma. Bu eksik, otomatik görevde ve çıktı raporunda gizlenmez.
