# Muhtemel Aşk: Colab altyazı akışı

RunPod ve açık bir PC gerektirmez. ASR, Colab GPU'da çalışır. Çeviri mevcut
ChatGPT ZIP sözleşmesini kullanır. GPU ile gerçek bölüm denemesi henüz yapılmamış
bir üretim adayıdır; test başarısı ses/çeviri kalitesinin kanıtı değildir.

## Telefonda kullanım

1. `Muhtemel_Ask.ipynb` dosyasını GitHub'dan indirip Colab'da **Upload notebook**
   ile aç. Özel repo için Colab'ın GitHub erişimi zaten açıksa doğrudan da açabilirsin.
   Notebook kodu ve sözlüğü içinde taşır; çalışma sırasında GitHub token'ı istemez.
2. **Runtime > Change runtime type > GPU** seç. `EPISODE = 15`, `MODE = "prepare"`.
3. `SOURCE_URL` tam bölüm bağlantısıdır. Boşsa resmî kanalda yalnız tam bölüm adı
   aranır. Fragman veya en son rastgele video seçilmez. İndirme engellenirse kaynak
   dosyayı Drive'a koyup `SOURCE_FILE` alanını kullan. Engeli aşma döngüsü yoktur.
4. Hazırlama hücrelerini sırayla çalıştır; Drive bağlama iznini ver.
5. Gösterilen `TRANSLATION_PACK.zip` dosyasını ChatGPT'ye ver. Paket içindeki
   talimatlara göre tüm batch'leri çevirmesini iste. GPU oturumunu beklerken kapat.
6. Dönüş ZIP'ini aynı `handoff` klasörüne koy. `MODE = "finish"` seç. Bu aşama
   CPU oturumunda da çalışır. Kurulum ve işlem hücrelerini yeniden çalıştır.
7. TR ve ID SRT'ler ile sorun listesi oluşur. **Dinleme ve düzeltme** hücresini
   aç. İşaretli satırları, örnekleri ve olası eksik konuşmaları dinle. Önizlemede
   yazı yalnız seçili zaman aralığında görünür. Zaman alanları mutlak milisaniyedir.
8. SRT'yi kaynak videoyla oynatıp bölümü kontrol et. İstenirse hızlı MKV önizlemesi
   veya CPU'da daha yavaş altyazısı gömülü MP4 üretilebilir.

Çeviri için kullanılacak kısa istek:

> Bu paketin TRANSLATION_INSTRUCTIONS.md, EVIDENCE_NOTES.md, manifest ve
> sözlüğüne uy. Tüm batch'leri doğal Endonezceye çevir, Türkçe ASR hatalarını
> yalnız kanıtın desteklediği ölçüde düzelt. Kimlik/sıra/zamanları değiştirme.
> Belirsizliği işaretle. Yalnız belirtilen dönüş ZIP'ini ver.

## Zamanlama yaklaşımı

- `large-v3` orijinal sesten kelime zamanlarını üretir. Whisper'ın kendi attention /
  DTW kelime zamanları kullanılır; düzeltilmiş metne ikinci CTC/WhisperX hizalaması yoktur.
- Cue başlangıcı ilk kelimenin başlangıcıdır. Sona en fazla 200 ms boşluk eklenir.
  Bir sonraki konuşma bu boşluğu sınırlar. Konuşmanın kendisi çakışmayı gizlemek için kesilmez.
- 450 ms ve üzeri durak, ASR segment sınırı, noktalama, 5 saniye veya 64 kaynak
  karakteri yeni cue açabilir. Her kelime tam bir cue'ya aittir.
- 42 karakter/satır, en fazla 2 satır ve 20 karakter/saniye sınırı çeviri sonrası
  kontrol edilir. Uzun metin otomatik kesilmez ve süreyi uzatarak gizlenmez.
- Düşük güven, şüpheli uzun/sıfır kelime zamanı, VAD'a göre erken başlangıç,
  örtüşen konuşma, sürekli konuşmada parça sınırı ve olası eksik konuşma incelemeye girer.
  VAD ne sözcük zamanının ne de konuşma eksiksizliğinin kesin kanıtıdır.
- İsimsiz konuşmacılar için sahte speaker kimliği üretilmez. Üst üste konuşma ve
  müzik altındaki konuşma insan kulağıyla kontrol edilmelidir.

## Dosyalar ve kesintiden devam

Varsayılan klasör:
`MyDrive/Muhtemel_Ask_Subtitles/Colab_v1/Muhtemel Ask 15.Bolum/`

| Dosya | İşlev |
|---|---|
| `source.media`, `source.json` | Değişmez kaynak ve SHA-256 |
| `audio.wav`, `audio.json` | Medya zaman eksenine oturtulmuş mono ses |
| `asr_plan.json` | Model revision, paket sürümleri, VAD ve parça planı |
| `asr/chunk_*.json` | Yaklaşık 5 dakikalık tamamlanan ASR parçaları |
| `schema.json`, `manifest.json` | Kelime kimliği, cue zamanı ve çeviri sözleşmesi |
| `handoff/` | Giden ve gelen tek çeviri ZIP'i |
| `review.json` | Kaynak ASR'yi değiştirmeyen dinleme kararları |
| `output/<result-hash>/` | SRT, QA, isteğe bağlı video |
| `latest_output.json` | En son çıktının açık durumu |

Kesintide tamamlanmış parça tekrar hesaplanmaz. Yalnız yarım kalan parça yeniden
çalışır. Yeni kaynak veya değişmiş checkpoint sessizce kabul edilmez. Sürüm/hash
uyuşmazlığında eski kayıtlar silinmez; farklı bir çalışma klasörü kullanılır.

Sorun varken `.draft.srt` hemen kullanılabilir ama durum `DRAFT_REVIEW_REQUIRED`
kalır. Yapısal kontroller ve dinleme kararları tamamlanınca `REVIEWED` olur.
Bu etiket bir modelin akustik/semantik doğruluk sertifikası değildir.

Olası eksik konuşma uzunsa dinleme aracındaki JSON alanına birden fazla ayrı cue
(start_ms, end_ms, tr_final, id_final) girilebilir. Müzik veya yanlış VAD tespiti
olduğu dinlenerek doğrulanırsa gerekçe ile işaretlenebilir. Konuşma otomatik silinmez.

Drive'a yazma sonrası boyut/SHA-256 okuma kontrolü vardır. Bu, bağlı dosya sistemi
üzerinden readback'tir; bağımsız Drive API indirmesiyle teslim doğrulaması değildir.

## Cuma 06:00 sınırı

Bu notebook bir zamanlayıcı değildir. Telefonda oturumu başlatmak ve Drive izni
vermek gerekir. Colab kaynakları/kotaları değişebilir ve oturum kapanabilir. PC
kapalı olabilir; fakat gözetimsiz, garantili 06:00 başlangıcı bu değişiklikle kurulmaz.
ChatGPT çeviri ZIP'inin gönderilmesi/geri konması da bir kullanıcı adımıdır.

Resmî kaynaklar:
- https://research.google.com/colaboratory/faq.html
- https://github.com/SYSTRAN/faster-whisper#word-level-timestamps
- https://github.com/SYSTRAN/faster-whisper#gpu
- https://github.com/yt-dlp/yt-dlp/wiki/EJS

## Geliştirici kontrolü

```bash
PYTHONPATH=src python -m pytest tests/test_colab_flow.py -q
python colab/build_notebook.py
git diff --check
```

`build_notebook.py`, bağımsız modülü ve mevcut üç dil dosyasını notebook içine
birebir yerleştirir. Hücrelerde çıktı, anahtar veya kullanıcıya ait medya yoktur.
