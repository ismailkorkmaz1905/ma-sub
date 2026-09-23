# Muhtemel Aşk: Colab altyazı akışı

RunPod ve açık bir PC gerektirmez. ASR, Colab GPU'da çalışır. Çeviri mevcut
ChatGPT ZIP sözleşmesini kullanır. Gerçek Colab denemesinin güncel sonucu `../automation/README.md` içindedir;
yazılım testi ses/çeviri kalitesinin kanıtı değildir.

## Telefonda kullanım

1. `Muhtemel_Ask.ipynb` dosyasını GitHub'dan indirip Colab'da **Upload notebook**
   ile aç. Özel repo için Colab'ın GitHub erişimi zaten açıksa doğrudan da açabilirsin.
   Notebook kodu ve sözlüğü içinde taşır; çalışma sırasında GitHub token'ı istemez.
2. **Runtime > Change runtime type > L4 GPU** seç (yoksa T4). `EPISODE = 15`, `MODE = "prepare"`.
3. `SOURCE_URL` tam bölüm bağlantısıdır. Boşsa Show TV sayfasındaki gerçek MP4
   bulunur. Resmî altyazı hazır değilse beklenmeden GPU ASR başlar. İndirme engellenirse kaynak
   dosyayı Drive'a koyup `SOURCE_FILE` alanını kullan. Engeli aşma döngüsü yoktur.
4. Hazırlama hücrelerini sırayla çalıştır; Drive bağlama iznini ver.
5. Gösterilen `TRANSLATION_PACK.zip` dosyasını ChatGPT'ye ver. Paket içindeki
   talimatlara göre tüm batch'leri çevirmesini iste. İşlem hücresi Drive kaydını tamamlayıp GPU’yu kendisi kapatır.
6. Dönüş ZIP'ini bölüm kökündeki `handoff` klasörüne koy. `MODE = "finish"` seç.
   Yeni L4/T4 GPU oturumunda kurulum ve işlem hücrelerini yeniden çalıştır; MP4 otomatik gömülür.
7. TR ve ID SRT'ler ile sorun listesi oluşur. **Dinleme ve düzeltme** hücresini
   aç. İşaretli satırları, örnekleri ve olası eksik konuşmaları dinle. Önizlemede
   yazı yalnız seçili zaman aralığında görünür. Zaman alanları mutlak milisaniyedir.
8. Altyazısı gömülü MP4'ü oynatıp kontrol et. Eski projenin Arial stili ve NVIDIA
   kodu kullanılır. Sorunlar veya dinleme eksikleri varsa video draft olarak kalır.

Çeviri için kullanılacak kısa istek:

> Bu paketin TRANSLATION_INSTRUCTIONS.md, EVIDENCE_NOTES.md, manifest ve
> sözlüğüne uy. Tüm batch'leri doğal Endonezceye çevir, Türkçe ASR hatalarını
> yalnız kanıtın desteklediği ölçüde düzelt. Kimlik/sıra/zamanları değiştirme.
> Belirsizliği işaretle. Yalnız belirtilen dönüş ZIP'ini ver.

## Zamanlama yaklaşımı

- `large-v3` önce geniş ses aralığını bağlam ve olası eksik konuşma için çözer.
  Ardından konuşma aralıklarını ayrı ayrı çözer; sessizlikleri yapıştırarak kelime
  zamanı üretmez. Geniş ASR metni ek metin kanıtı olarak pakete eklenir.
  Whisper attention/DTW kullanılır; düzeltilmiş metne CTC/WhisperX hizalaması yoktur.
- Cue başlangıcı ilk kelimenin başlangıcıdır. Sona en fazla 200 ms boşluk eklenir.
  Bir sonraki konuşma bu boşluğu sınırlar. Konuşmanın kendisi çakışmayı gizlemek için kesilmez.
- 450 ms ve üzeri durak, ASR segment sınırı, noktalama, 6 saniye veya 84 kaynak
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
| `source.mp4`, `source.json` | Değişmez kaynak ve SHA-256 |
| `audio.wav`, `audio.json` | Medya zaman eksenine oturtulmuş mono ses |
| `asr_plan.json` | Model revision, paket sürümleri, VAD ve parça planı |
| `asr/chunk_*.json` | Yaklaşık 5 dakikalık tamamlanan ASR parçaları |
| `schema.json`, `manifest.json` | Kelime kimliği, cue zamanı ve çeviri sözleşmesi |
| `handoff/` | Giden ve gelen tek çeviri ZIP'i |
| `review.json` | Kaynak ASR'yi değiştirmeyen dinleme kararları |
| `output/<result-hash>/` | SRT, QA, altyazısı gömülü MP4 |
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

## Cuma 06:00 Singapur görevi

ChatGPT kontrol görevi kuruludur; ayrıntı ve gerçek kurulum durumu
[otomasyon notlarında](../automation/README.md). Notebook tek başına zamanlayıcı
değildir. GPU tahsisi, oturum ve Drive yetkisi başlangıçta kontrol edilmelidir.
ChatGPT çeviri paketini Drive üzerinden kendisi işler; kullanıcıdan dosya taşıması
istenmez. Colab erişimi henüz doğrulanmamış bir çalışmayı otomatik başarı saymaz.

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

## Boyut ve GPU bütçesi

Varsayılan hedef 4.8 GB, kesin teslim sınırı 5,000,000,000 byte altıdır. Gerçek
çıktı büyükse yalnız bir küçültme denemesi yapılır; video kesilmez. İki encode
için ortak üst süre bir saattir. Kaynak çözünürlüğü korunur.

`AUTO_RELEASE_GPU=True` prepare/finish sonunda ve işlem hatasında Drive flush
başarılı olunca oturumu kapatır. Çeviri sırasında GPU beklemez. Dinleme ekranını
kullanmak için yeniden bağlanıp kurulum hücrelerini çalıştır; yalnız inceleme
sırasında otomatik kapanmayı kapatabilirsin. Kurulum veya Drive bağlantı hatası
olursa çalışan GPU'yu ayrıca kapat.

Sıfır süreli kelime içeren ve geniş ASR tarafından desteklenmeyen belirli stok
sözler, ham kanıtları korunarak incelemeye ayrılır. Bunların ses aralıkları QA'da
kalır; konuşma yokmuş gibi sessizce onaylanmaz.
