# Muhtemel Ask Subtitles

Tek üretim komutu vardır:

```powershell
.\mas.ps1 run 14
```

Aynı komut yeni bölümü başlatır, yarım kalan bölümü sürdürür ve gereken ZIP
dönüşünü bekler. Yeni kaynak için URL açıkça verilebilir:

```powershell
.\mas.ps1 run 15 --source-url 'SOURCE_URL'
```

## Günlük kullanım

```powershell
.\mas.ps1 doctor --controller
.\mas.ps1 status 14
.\mas.ps1 status 14 --json
```

Normal `status` yalnız mevcut aşamayı, kaynak dosyayı, bekleyen ChatGPT ZIP'ini
ve final çıktıyı gösterir. `--json` ayrıntılı teknik kaydı verir.

Bir bölümde operatörün ilgilendiği yerler:

```text
source/               değişmez kaynak video
translation_input/    ChatGPT'ye verilecek ZIP
translation_output/   ChatGPT'den dönen ZIP
final/                altyazı ve video çıktıları
```

`prepare/`, `review/`, `parts/`, `work/` ve `.mas/` otomatik iç kayıtlardır.
Normal kullanımda açılmaları veya düzenlenmeleri gerekmez. Her çalıştırma yalnız
`.mas/run.log` dosyasını yeniler; tarih damgalı sınırsız log üretmez.

## Akış

EP15 ve sonrası varsayılan olarak `semantic-block-v1` kullanır:

```text
kaynak -> GPU Türkçe ASR -> kelime zaman çizgisi -> gerekirse semantik ZIP
       -> Endonezce ZIP -> altyazı QA -> MP4 -> Drive hash doğrulaması
```

EP14 ve önceki kayıtlar mevcut politikalarını korur. `strict-ctc-v1`,
delivery-first ve emergency çıktıları birbirinin PASS kaydı olamaz.

## Değişmez kurallar

- Kaynak SHA-256 kaydından sonra değişmez.
- GPU aşamaları sessizce CPU'ya düşmez.
- ChatGPT metni düzeltebilir; zaman ve kimlik alanlarını değiştiremez.
- Farklı konuşmacıların metni birleştirilmez.
- Drive teslimi ancak byte sayısı ve SHA-256 readback eşleşince PASS olur.
- Test çalıştırmak ücretli RunPod çalıştırma izni değildir.

## Kod

Üretim kodu `src/mas/`, ayarlar `config/`, testler `tests/` altındadır.
`legacy/` yalnız referanstır.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q tests
git diff --check
```

Güncel teknik belgeler:

- [Mimari kararlar](docs/ARCHITECTURE_DECISIONS.md)
- [EP15 semantik akış](docs/SEMANTIC_ALIGNMENT_EP15.md)
- [EP14 çalıştırma sınırı](docs/EP14_READY_2026-09-18.md)
- [Delivery-first](docs/DELIVERY_FIRST_2026-09-15.md)
- [Codex handoff](docs/CODEX_HANDOFF.md)
- [Astra handoff](docs/ASTRA_HANDOFF.md)
