# Muhtemel Ask Subtitles

Tek komut, tek çeviri paketi, dört klasör.

```powershell
.\mas.ps1 run 15 --source-url "VIDEO_URL"
```

Elinde video varsa:

```powershell
.\mas.ps1 run 15 --source "D:\video.mp4"
```

Program GPU ile Türkçe ASR yapar ve burada durur:

```text
handoff/Muhtemel Ask 15.Bolum_TRANSLATION_PACK.zip
```

ZIP'i ChatGPT'ye ver. ZIP'in içindeki `INSTRUCTIONS.md` dönüş biçimini anlatır.
Dönen dosyayı şu adla aynı klasöre koy:

```text
handoff/Muhtemel Ask 15.Bolum_TRANSLATED.zip
```

Sonra aynı komutu tekrar çalıştır:

```powershell
.\mas.ps1 run 15
```

Türkçe SRT, Endonezce SRT ve Endonezce gömülü MP4 `output/` içine yazılır.

## Klasörler

```text
source/   değişmeyen kaynak video
work/     state.json, ASR blokları ve doğrulanmış çeviri
handoff/  ChatGPT'ye giden ve dönen iki ZIP
output/   iki SRT, MP4 ve varsa Drive makbuzu
```

Tarih damgalı log, mail kuyruğu, review ağacı, partial/whole ağacı ve eski bölüm
özel durumları yoktur.

## Akış

```text
video -> faster-whisper GPU ASR -> tek ChatGPT ZIP'i
      -> kimlik/sıra doğrulaması -> TR + ID SRT -> gömülü MP4
```

ChatGPT yalnız Türkçe ve Endonezce metni değiştirir. Blok kimliği, sırası ve
zamanı Python'da sabit kalır. Eski veya eksik dönüş reddedilir. Kaynak videonun
SHA-256 değeri değişirse çalışma durur.

Drive teslimi istenirse `MAS_DRIVE_REMOTE` ayarlanır. Program MP4'ü yükler,
yeniden indirir ve byte sayısı ile SHA-256 eşleşmeden `DONE_DRIVE` yazmaz.
Değişken yoksa sonuç `DONE_LOCAL` olur.

```powershell
.\mas.ps1 status 15
.\mas.ps1 doctor
.\mas.ps1 test
```

`--subtitles-only` MP4 üretmeden yalnız iki SRT'yi tamamlar.

## Kurulum

Python 3.11, FFmpeg ve CUDA destekli NVIDIA GPU gerekir.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.lock
```

RunPod veya başka bir GPU makinesi kullanılabilir; depo Pod satın almaz ya da
controller çalıştırmaz. GPU işi bitip çeviri ZIP'i beklendiğinde Pod'u kapat.
