# Muhtemel Ask Subtitles

## PC kapalıyken, RunPod olmadan

[Yeni akış](automation/README.md): bölüm videosunu Drive'a indir → resmî altyazı
hazırsa kullan, yoksa GPU ASR → mevcut ChatGPT ile doğal Endonezce → **altyazısı
gömülü MP4**. Resmî altyazının ertesi gün çıkması beklenmez.

[Colab notebook](colab/Muhtemel_Ask.ipynb), [kullanım](colab/README.md).
`video_flow.py` eski `engine/burned_mp4.py` NVIDIA kodunu kullanır. Düzeltilmiş
metne yeniden forced alignment uygulanmaz. Canlı denemenin kapsamı ve otomasyonun
kurulum durumu `automation/README.md` içindedir.

## Mevcut üretim komutu

Bir bölümü başlatmak veya kaldığı yerden sürdürmek için tek komut kullanılır:

```powershell
.\mas.ps1 run 15 --source-url 'SOURCE_URL'
```

Sonraki çalıştırmalarda URL gerekmez:

```powershell
.\mas.ps1 run 15
```

Durumu görmek için:

```powershell
.\mas.ps1 status 15
```

## Bölüm klasörü

```text
source/    değişmeyen kaynak video
handoff/   ChatGPT'ye verilen ve geri alınan ZIP dosyaları
output/    altyazı, video ve teslim makbuzları
work/      otomatik ara dosyalar
.mas/      tek güncel çalışma logu
parts/     yalnız parçalı teslim gerekirse oluşur
```

Normal kullanımda yalnız `handoff/` ve `output/` ile ilgilenilir. Her çalıştırma
`.mas/run.log` dosyasını yeniler; tarih damgalı log yığını yoktur.

## Akış

```text
kaynak -> GPU Türkçe ASR -> semantik zamanlama -> Endonezce çeviri
       -> altyazı kontrolü -> MP4 -> Drive byte ve SHA-256 doğrulaması
```

Bir ZIP dönüşü gerektiğinde komut durur, tam dosya yolunu gösterir ve RunPod'u
güvenli biçimde kapatır. Dosya yerine konduktan sonra aynı `run` komutu sürdürür.

## Değişmez kurallar

- Kaynak SHA-256 kaydından sonra değişmez.
- GPU aşamaları CPU'ya düşmez.
- Metin, zaman ve kimlik alanlarının yetkileri birbirinden ayrıdır.
- Drive teslimi ancak byte sayısı ve SHA-256 readback eşleşince tamamlanır.
- Test veya kod düzenleme ücretli RunPod çalıştırma izni değildir.

Kod `src/mas/`, ayarlar `config/`, testler `tests/` altındadır.

```powershell
.\mas.ps1 doctor --controller
& .\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Teknik ayrıntı gerektiğinde [mimari kararlar](docs/ARCHITECTURE_DECISIONS.md) ve
[semantik zamanlama sözleşmesi](docs/SEMANTIC_ALIGNMENT.md) okunur.
