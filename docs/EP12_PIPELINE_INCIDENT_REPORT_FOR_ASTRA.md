# Muhtemel Aşk Bölüm 12 — Pipeline Olay ve İyileştirme Raporu

**Kapsam:** 4–5 Eylül 2026 Bölüm 12 altyazı üretimi  
**Saat dilimi:** Tüm saatler Singapore Time — SGT (UTC+8)  
**Amaç:** Astra'nın kalıcı, tek komutla çalışan altyazı pipeline'ını inşa ederken Bölüm 12'de yaşanan gecikmeleri ve veri bütünlüğü hatalarını tekrar etmemesi

## 1. Yönetici özeti

Bölüm 12'nin kaynak videosunun Drive'a alınmasından final Endonezce SRT'nin Drive'a yazılmasına kadar gözlenen toplam pencere **4 Eylül 07:42:08–5 Eylül 05:34:09**, yani yaklaşık **21 saat 52 dakika** sürdü. Bunun tamamı aktif GPU süresi değildi; hazırlık, kullanıcı/araç geçişleri, Colab GPU kotası, doğrulayıcı hataları, manuel patch döngüleri ve dosya aktarımındaki beklemeler bu pencereye dahildir.

Pratik final çıktı üretildi ve Drive'a hash doğrulamasıyla yazıldı:

- Dosya: `final/subtitles/Muhtemel Ask 12.Bolum-id.srt`
- Drive dosya kimliği: `1z6nHOdwzuyAG67Gr9s2XRk7qn1Fm4AQX`
- Drive oluşturulma zamanı: **5 Eylül 05:34:09.686 SGT**
- Boyut: **201.051 bayt**
- SHA-256: `da991f8e46cc71abda6bcb6d4e42300e4285cbd2033d6a01bb08bbd42baa1fd4`
- Süre: `00:00:05,370–02:16:16,800`
- Toplam cue: **3.346**
- Boş cue: **0**
- Sıfır/negatif süreli cue: **0**
- En uzun satır: **42 karakter**

Ancak kritik ayrım şudur: Bu dosya **pratik emergency/recovery finalidir**. Resmî `SYSTEM_V2_BETA` zincirindeki exact `ID_TRANSLATION_PACK.zip` PASS alınmadı, exact `ID_TRANSLATED.zip` üretilmedi ve `03_FINALIZE_V2.ipynb` resmî olarak PASS vermedi. Bu nedenle bu sonucu formal V2 release artifact'i veya sıfır-hata garantisi olarak kullanmamak gerekir.

## 2. Zaman kanıtı sınıfları

- **Kesin:** Google Drive `createdTime`/`modifiedTime`, final hash/byte readback veya araç tarafından ölçülen süre.
- **Çok güçlü:** Yerel dosya `mtime` kayıtları ve sıralı patch/log dosyaları.
- **Tahmini:** Konuşma sırası, görülen bakiye ve saatlik ücret üzerinden hesaplanan aralık.

Bu rapor kesin olmayan noktaları özellikle tahmini olarak işaretler.

## 3. Ayrıntılı zaman çizelgesi

| Tarih ve saat (SGT) | Süre | Aşama | Gerçekte ne oldu | Kanıt |
|---|---:|---|---|---|
| 4 Eyl 07:42:07.956–07:42:33.936 | 26 sn | Kaynak video | `Muhtemel Ask 12.Bolum.mkv` Drive'a yazıldı; boyut 1.035.830.770 bayt. Metadata ve download marker oluştu. | Kesin Drive zamanı |
| 07:42:49.516–07:43:48.667 | 59 sn | Ses çıkarma | `audio.flac` üretildi; 323.891.991 bayt. Audio metadata ve done marker oluştu. | Kesin Drive zamanı |
| 07:43:49–09:59:26 | 2 sa 15 dk 37 sn | PREPARE/ASR ilk bölüm | Whisper/VAD hazırlığı ilerledi; ilk speech-hole klasörü 09:59'da oluştu. Bu aralık kesintisiz compute olarak doğrulanamaz. | Drive artifact aralığı |
| 09:59:26–10:01:13 | 1 dk 47 sn | Speech-hole paketleme | `speech_holes` ve `speech_hole_audio` klasörleri oluştu. | Kesin Drive zamanı |
| 10:01:13–11:48:13 | 1 sa 47 dk | ASR devam/checkpoint | Uzun ASR koşusu; `raw_asr_v2.recovery.json` 11:48'de yazıldı. | Kesin artifact zamanı |
| 11:48:13–12:37:44 | 49 dk 31 sn | ASR tamamlama | ASR devam etti; `asr_hallucination_audio` klasörü 12:37:44'te oluştu. | Kesin artifact zamanı |
| 12:37:44–12:38:14 | 30 sn | Raw ASR finalize | `raw_asr_v2.json` ve marker üretildi. Raw JSON 13.440.840 bayt. | Kesin Drive zamanı |
| 12:38:14–12:56:45 | 18 dk 31 sn | TR correction pack | Exact `Muhtemel Ask 12.Bolum_TR_CORRECTION_PACK.zip` üretildi; 28.114.783 bayt. | Kesin Drive zamanı |
| 12:56:45–15:28 | 2 sa 31 dk | Handoff/bekleme | Paket hazırdı; metin düzeltme konuşması yaklaşık 15:28'de başladı. Bu pencerenin tamamı aktif işlem değildir. | Proje konuşma zamanı + Drive |
| 15:28–16:03:32 | yaklaşık 35 dk | Türkçe metin düzeltme | 2.977 Türkçe kayıt işlendi; `TR_TEXT_CORRECTED.zip` ilk kez 16:03:32'de oluştu. | Konuşma başlangıcı + Drive create |
| 16:03:32–17:23:01 | 1 sa 19 dk | Girdi/hash değişikliği sonrası onarım | Raw ASR/source binding tarafında değişiklik oldu; eski correction sonucu yeni hash zincirine uymadı. | Drive modified zamanları |
| 17:23:01–17:24:28 | 1 dk 27 sn | Toplu rebind | `source.media`, raw ASR marker, raw ASR, correction pack ve TR corrected ZIP sırayla yeniden bağlandı. | Kesin Drive modified zamanları |
| 17:34:22–18:51:26 | 1 sa 17 dk 04 sn | Colab CPU audio review | `audio_review_v2.recovery.json` yaşadı ve güncellendi. GPU kotası olmadığı için CPU koşusu çok yavaştı; mevcut checkpoint 1–9/352 civarından resume edildi ve eski malformed-word arıza noktası geçildi. | Drive recovery zamanı + çalışma logu |
| yaklaşık 18:51–19:20 | yaklaşık 29 dk | Colab GPU ve bulut geçişi | Eski Colab session'ları kapatıldı, T4 seçildi fakat GPU quota reddetti. RunPod kayıt/Cloudflare/ödeme ve Pod kurulumu yapıldı. | Konuşma sırası; dakika tahmini |
| yaklaşık 19:18 | — | RunPod başlangıcı | RTX 4090 Pod yaklaşık bu saatte ücretlenmeye başladı. Saatlik fiyat `$0.74`; kesin başlangıç billing export'u yoktur. | `$10.00→$2.40` bakiye ve `$0.74/saat` üzerinden çıkarım |
| yaklaşık 19:20–21:18 | yaklaşık 1 sa 58 dk | RunPod bootstrap + aktarım + acil ilk 40 | 412.702.720 baytlık bundle hazırlandı. Port 8000 açık olmadığı için ilk aktarım 404 verdi; yalnız 8888/Jupyter açıktı. İlk 40 dakikalık acil ID SRT üretildi. | Çalışma logu + Drive zamanı |
| 21:18:08–21:20:02 | 1 dk 54 sn | İlk 40 teslimi | `Muhtemel_Ask_12.Bolum-id-FIRST40.srt` ve `Muhtemel_Ask_12.Bolum_FIRST40_ID.zip` Drive'a yazıldı. | Kesin Drive zamanı |
| 21:19:40 | — | İlk 40 yerel QA | 843 cue'luk SRT ve QA üretildi. 858 kaynak kaydın 13'ü non-dialogue blank, 2'si reviewed discard; 29 ayrı kaynak-overlap vardı. | Çok güçlü yerel mtime + QA JSON |
| 21:27:19 | — | Source görünürlüğü onarımı | İlk 40 SRT'nin bir kopyası `source/Muhtemel Ask 12.Bolum.id.srt` olarak yazıldı. | Kesin Drive zamanı |
| 22:16:47 | — | Kaynak dosya görünürlüğü onarımı | Kullanıcının “source klasörü kayboldu” uyarısından sonra 1.035.830.770 baytlık MKV episode root'ta tekrar görünür hâle getirildi/kopyalandı. | Kesin Drive zamanı |
| 22:35:18–00:45:49 | 2 sa 10 dk 31 sn | Acoustic patch döngüsü | `review_98_113`, `patch_acoustic8…21`, `make_safe3/4`, focused GPU review ve timing override dosyaları art arda üretildi. | Çok güçlü yerel mtime zinciri |
| 00:45:49–yaklaşık 04:05 | yaklaşık 3 sa 19 dk | Derin align/VAD teşhisi | Custom runners 67–75; overlap, VAD coverage, reviewed-dialogue padding ve stale review-window hataları incelendi. Exact bitiş zamanları yalnız RunPod içindeydi ve Pod kapatılırken container state korunmadı. | İşlem çıktıları; zaman sınırları tahmini |
| yaklaşık 04:05–04:10 | yaklaşık 5 dk | Recovery Endonezce üretimi | NLLB 1.3B yerel GPU modeli hazırlandı. Model yükleme **92,0 sn**, 1.897 benzersiz metnin batch çevirisi **15,3 sn** sürdü. 2.070 tail segmentinden 2.503 cue üretildi. | Kesin model log süreleri; mutlak saat yaklaşık |
| yaklaşık 04:11–05:28 | yaklaşık 1 sa 17 dk | Dosya aktarım kilitlenmesi | Jupyter file chooser/upload çağrısı **4.601,2 sn = 1 sa 16 dk 41,2 sn** takıldı. GPU bu sırada fiilen boş olmasına rağmen Pod ücret yazmaya devam etti. | Kesin araç wall-time |
| yaklaşık 05:28–05:34 | yaklaşık 6 dk | Alternatif export + merge + QA | Tail, Jupyter text editor üzerinden tam clipboard kopyasıyla alındı; SHA eşleşti. İlk 40 + tail birleştirildi, cue'lar global zaman sırasına kondu ve QA yapıldı. | İşlem logu + final hash |
| 05:34:00.108 | — | Final Drive klasörü | `final/subtitles` klasörü oluşturuldu. | Kesin Drive zamanı |
| 05:34:09.686 | — | Final upload | `Muhtemel Ask 12.Bolum-id.srt` Drive'a yazıldı. | Kesin Drive zamanı |
| yaklaşık 05:34–05:36 | yaklaşık 2 dk | Readback ve shutdown | Drive raw bytes geri okundu; 201.051 bayt ve SHA eşleşti. RunPod `Not running`, `$0.00/hr`; bakiye `$2.40`. | Kesin readback; shutdown dakikası yaklaşık |

## 4. Sürelerin toplu özeti

| Büyük bölüm | Yaklaşık süre | Yorum |
|---|---:|---|
| Source + audio + raw ASR + correction pack | 5 sa 14 dk 38 sn | 07:42:08–12:56:45 artifact penceresi; kesintisiz compute olduğu iddia edilmemeli. |
| TR correction ilk üretim + hash rebinding | 1 sa 56 dk | 15:28–17:24; asıl sorun sonradan değişen input/hash zinciri. |
| Colab CPU audio-review denemesi | 1 sa 17 dk | Çok düşük ilerleme; GPU yoksa bu aşama CPU'ya otomatik düşmemeli. |
| Colab→RunPod geçişi ve ilk 40 | yaklaşık 2 sa 27 dk | Hesap, Pod, bundle, Jupyter portu ve acil çıktı. |
| Recorded acoustic patch döngüsü | 2 sa 10 dk 31 sn | 22:35–00:45; çok sayıda notebook monkeypatch'i. |
| Sonraki VAD/align araştırması | yaklaşık 3 sa 19 dk | 00:46–04:05; resmî pipeline PASS'e ulaşılamadı. |
| Gerçek final GPU çevirisi | 15,3 sn | Darboğaz çeviri compute'u değildi. |
| Model hazırlama | 92 sn | Kalıcı cache olursa sonraki bölümlerde sıfıra yakın olmalı. |
| Son dosya aktarımında boşa bekleme | 1 sa 16 dk 41 sn | En net önlenebilir kayıplardan biri. |
| Merge + Drive upload + hash verification | yaklaşık 6 dk | Otomatik olmalı ve sonunda compute hemen kapanmalı. |

## 5. RunPod kullanım ve maliyet özeti

- Pod: `muhtemel-ask-ep12`
- Pod ID: `p54vvbyu76eztn`
- GPU: RTX 4090, 24 GB VRAM
- CPU/RAM: 8 vCPU, yaklaşık 41–46 GB RAM
- Disk: 30 GB container + 50 GB network volume
- Saatlik compute: `$0.74/saat`
- Başlangıç kredi: `$10.00`
- Kapanış bakiye: `$2.40`
- Harcanan: yaklaşık `$7.60`
- Bu harcamanın `$0.74/saat` karşılığı: yaklaşık **10 saat 16 dakika** billable compute
- Buradan çıkarılan Pod başlangıcı: yaklaşık **4 Eylül 19:18 SGT**
- Kapanışta görülen durum: `Not running`, compute `Not running`, total `$0.00/hr`

En önemli maliyet dersi: GPU hesaplama kısa sürdü; kredi esas olarak setup, debug, UI aktarımı ve idle beklemede harcandı.

## 6. Teknik arıza zinciri

### 6.1 Colab provisioning kırılgandı

1. Temiz session kurulumu yaklaşık 10,4 GB paket/veri indirdi.
2. Drive mount ilave Google doğrulamasına girdi.
3. Colab GPU kotası T4 tahsisini reddetti.
4. Notebook `auto` ile CPU'ya düştüğünde 352 audio-review klibi pratik olmayacak kadar yavaşladı.
5. Eski session'ları terminate etmek GPU kotasını açmadı.

**Kural:** GPU zorunlu aşama CPU'ya sessizce düşmemeli. `require_gpu=true` ise 60 saniye içinde fail etmeli ve otomatik ikinci sağlayıcıya geçmeli.

### 6.2 Checkpoint vardı fakat stage bütünlüğü eksikti

- `audio_review_v2.recovery.json` korunabildi ve 1–9/352 kararları resume edildi.
- Ancak runtime, model cache, exact code commit ve input artifact set'i tek bir run manifestinde bağlı değildi.
- Input hash zinciri değişince correction pack ve corrected ZIP yeniden bağlanmak zorunda kaldı.

**Kural:** Her stage marker şu alanları taşımalı: `run_id`, git commit, container digest, input artifact SHA'ları, output SHA'ları, record count, completed UID set, start/end time, provider/GPU ve exact config.

### 6.3 RunPod'a veri transferi yanlış yüzeyden yapıldı

- Bundle: **412.702.720 bayt**.
- Port 8000 bekleniyordu fakat açık değildi; 404 alındı.
- Yalnız port 8888/Jupyter açıktı.
- `/content/drive/...` path'i gerçek senkron Google Drive mount'u gibi göründü fakat son yazılan tail Drive API listesinde görünmedi. Yani path varlığı remote persistence kanıtı değildi.
- Final dosya Jupyter editor + clipboard ile çekilip yerelden Drive connector'a yüklendi.

**Kural:** Production transfer yalnız `rclone`/Drive API/OBJECT storage ile yapılmalı. Başarı kriteri path'in var olması değil, remote metadata readback + byte size + SHA-256 eşleşmesidir.

### 6.4 Alignment validator aynı anda konuşan karakterleri yanlış modelledi

V2 doğrulayıcı birçok yerde tüm kelimeleri tek global non-overlap zinciri gibi ele aldı. Aynı anda konuşan iki ayrı karakterin kelimeleri zaman olarak kesişince bunlar yanlış biçimde unsafe/fail oldu.

**Doğru model:**

- Her utterance/speaker lane bağımsızdır.
- Aynı speaker içindeki kelimeler monoton olmalıdır.
- Farklı speaker/utterance'lar overlap edebilir.
- Overlap asla iki karakterin metnini tek cue'ya birleştirme gerekçesi değildir.
- Render katmanı ayrı cue'ları korumalı; gerekirse position/lane metadata üretmelidir.

### 6.5 Reviewed-dialogue window ile gerçek konuşma sınırı karıştırıldı

Audio-review klipleri bağlam padding'i içeriyordu. Validator, klibin tüm padded penceresini konuşmayla doldurulması gereken alan gibi değerlendirdi. Bu yüzden gerçek kelimeler doğru olsa bile confirmed-dialogue coverage fail verdi.

**Kural:**

- `clip_start/end`: dinleme bağlamı.
- `target_start/end`: karar verilen immutable hedef.
- `speech_start/end`: doğrulanmış gerçek konuşma.
- Coverage yalnız target/speech interval üzerinden değerlendirilmelidir; context padding üzerinden değil.

### 6.6 Stale override'lar metni ve zamanı bozdu

Notebook içinde timing düzeltmesi yapan override sözlükleri aynı zamanda `replacement_text` değiştiriyordu. Bazı override'lar eski correction sürümüne aitti. Bunun sonucu doğrulanmış uzun cümleler `Ha?`, `Ne?`, `Ya.` veya `Evet.` gibi kısa metinlerle ezildi.

Örnek kritik UID'ler:

| UID | Bozuk kısa sonuç | GPU/ASR ile geri kazanılan içerik özeti |
|---|---|---|
| `MA12-TR-b44aaef79127047b` | `Ya.` | “Ya Rabbim, sen yardım et Allah'ım, ne olur.” |
| `MA12-TR-f123d5936e04720f` | `Ha?` | Kadir Bey'in ne düşüneceğini merak eden tam cümle |
| `MA12-TR-4e5175627329e0f4` | `Ya.` | “Hadi kızım… doğrusunu anlatalım…” |
| `MA12-TR-c715c233530a6bff` | `Ne?` | Levent Bey/Kadir Bey proje sorusu |
| `MA12-TR-ae1bd74ad155663e` | `Evet.` | “İnanın bana, geleceksiniz.” |
| `MA12-TR-034bd51a80eba74b` | `Ya.` | “Sen yine kendini mi dövdün…” |
| `MA12-TR-6fdf171ba993cd03` | eksik orta parça | Boğaz'a nazır otelde kat kapatma cümlesi |
| `MA12-TR-2f9548ad34558bde` | `Ya.` | Araziye çökme ve söyleyememe cümlesi |
| `MA12-TR-83c4f51011911660` | `Evet.` | Hamza'nın adamlarının ofis çevresinde olması |
| `MA12-TR-53fb34c7b748a312` | eksik | “Yapmasaydın, onu yapmasak mı?” |

**Kural:** Timing düzeltmesi `tr_corrected` alanına asla yazamaz. Text correction ve timing correction ayrı schema/type olmalıdır. Her override exact `input_sha256 + UID + expected old text + expected old time` precondition'ına bağlanmalı; precondition tutmazsa fail etmelidir.

### 6.7 VAD/coverage sayıları gerçek sorunu görünür kıldı

Bir ara koşuda:

- unresolved speech regions: **94**
- unsafe words: **956**
- confirmed-dialogue coverage failures: **78**

VAD-fit düzeltmesinden sonra:

- unresolved: **15**
- unsafe words: **4**
- confirmed-dialogue failures: **5**

Bu ilerleme iyiydi; fakat kalanların bir kısmı gerçek hata değil, duplicate suffix, sessizlik, context padding veya farklı speaker overlap idi. Validator bu sınıfları açıkça temsil etmiyordu.

### 6.8 Son 15 unresolved interval için regression fixture gerekli

| Interval (ms) | Gerçek durum / ders |
|---|---|
| 97.860–98.012 | “İyi misin?” tekrarının duplicate suffix'i; ayrı yeni konuşma değil. |
| 173.360–173.468 | Önceki “Abi!” sonrasındaki duplicate kırıntı. |
| 288.920–289.340 | “Selma anne” ile “Nasılsın” arasında geçiş/sessizlik. |
| 326.160–326.620 | “Sevgilisini görsün, değil mi?”; stale override cümleyi kısaltmıştı. |
| 2.116.260–2.117.700 | Uzun Kadir Bey cümlesi yanlışlıkla `Ha?` olmuştu. |
| 2.187.860–2.189.060 | Levent/Kadir proje sorusu yanlışlıkla `Ne?` olmuştu. |
| 2.947.410–2.947.644 | Overlap/repeated “Tamam… söz”; mevcut cue'lar konuşmayı zaten temsil ediyor. |
| 3.814.724–3.815.772 | Parçalanmış “kendimi / aklımı, kalbimi” konuşması; cue'lar mevcut. |
| 5.145.703–5.147.510 | Sonraki speaker 5.145.403 civarında başlıyor; coarse boundary geç kalmış. |
| 5.218.820–5.219.640 | Çoğu sessizlik; “Hanım siz” yaklaşık 5.219.560'ta başlıyor. |
| 5.224.200–5.225.020 | Cue'lar arası sessizlik. |
| 6.596.440–6.596.604 | “Annemler nerede?” sonrasındaki sessizlik. |
| 6.612.490–6.612.860 | İki replik arasındaki sessizlik. |
| 7.806.564–7.806.812 | “Çıkalım” 7.806.264–7.806.664'e kadar uzuyor; source end geç güncellenmeli. |
| 8.006.968–8.007.420 | “Melis mi ya?” sonrasındaki sessizlik. |

Bu 15 olay unit/integration regression test olarak repoya konmalıdır.

### 6.9 Final export akışı otomatik değildi

- Jupyter file chooser 4.601,2 saniye kilitlendi.
- Compute idle iken ücret devam etti.
- Alternatif export, clipboard ve yerel merge gerekti.
- Başarılı upload sonrasında ayrı bir Drive raw readback ve SHA hesaplandı.

**Kural:** Finalizer dosyayı doğrudan Drive API'ye temp isimle yüklemeli, remote byte/hash doğrulamalı, exact isme atomic rename etmeli ve `finally` bloğunda compute shutdown çağırmalıdır.

## 7. Final recovery çıktısının yapısı ve sınırlamaları

### İlk 40 dakika

- 858 kaynak record
- 843 final cue
- 13 blank non-dialogue
- 2 reviewed discard
- 29 kaynak timing overlap
- Coverage: `00:00:05,370–00:40:00,000`
- Timing policy: immutable source intervals; bu parça formal forced-alignment PASS değildi.

### 40. dakikadan sonrası

- Forced-alignment segmenti: 2.910 toplam; tail'e giren 2.070
- Benzersiz Türkçe metin: 1.897
- Tail cue: 2.503
- Coverage: `00:40:00,001–02:16:16,800`
- Boş cue: 0
- Geçersiz timing: 0
- Max satır: 42 karakter

Tail'de uzun çeviriler okunabilirlik için birden fazla cue'ya bölündü. Bu pratik SRT için işe yaradı fakat official V2 prompt'taki “block bölme/birleştirme yok” kuralına aykırıdır. Strict pipeline, block kimliğini koruyarak `render_children` gibi açık bir alt-yapı tanımlamadıkça bu yöntemi kullanmamalıdır.

### Overlap yorumu

Final QA'da **102 ayrı overlapping cue pair** vardı; en yüksek overlap **3.620 ms**. Bunlar tek cue'ya birleştirilmedi. Ancak speaker ID olmadığı için yalnız zaman/UID ayrılığı korunabildi. Production için speaker lane veya diarization kimliği gereklidir.

### Translation yöntemi

İlk 40'ın mevcut Endonezce metni korundu. Kalan bölüm, RunPod üzerinde yerel NLLB 1.3B modeliyle batch çevrildi. Bu, acil recovery çözümüdür ve V2 talimatındaki “harici/model çevirisi yok; çeviriyi kendi dil yeteneğinle yap” kuralına uymaz. Strict release pipeline'ın çeviri aşaması exact pack + immutable field doğrulamasıyla ayrı yürümelidir.

## 8. Astra'nın inşa edeceği pipeline için zorunlu kurallar

### 8.1 Tek komutlu state machine

```text
INGEST
  -> PREPARE_TR
  -> TR_TEXT_CORRECTION
  -> AUDIO_REVIEW
  -> FORCED_ALIGN
  -> BUILD_ID_PACK
  -> ID_TRANSLATION
  -> FINALIZE
  -> UPLOAD_AND_READBACK_VERIFY
  -> SHUTDOWN_COMPUTE
```

Her stage `PENDING/RUNNING/PASS/FAIL/BLOCKED` durumuna sahip olmalı. Bir stage PASS olmadan sonraki stage başlayamaz. UI yalnız gerçekten doğrulanmış state'i göstermeli.

### 8.2 Strict ve emergency modları ayrılmalı

**Strict release mode**

- Exact V2 ZIP ve schema contract.
- UID/order/timing/tr_text immutable.
- No split/merge/reorder.
- Audio-review ve alignment hash-bound.
- Exact `ID_TRANSLATION_PACK.zip`, `ID_TRANSLATED.zip`, `03_FINALIZE` PASS.

**Emergency watch-now mode**

- İlk 40 veya full SRT hızlı üretilebilir.
- Çıktı adına `EMERGENCY`/`DRAFT` etiketi konur.
- Strict klasörünü veya marker'larını asla overwrite etmez.
- Kullanıcıya formal PASS olmadığı açıkça gösterilir.

### 8.3 Source hiçbir koşulda hareket ettirilmemeli

- `source/` immutable ve read-only kabul edilmeli.
- Copy/move/delete ayrı yetki ve dry-run gerektirmeli.
- `source manifest` en az iki bağımsız konumda saklanmalı.
- Her başlangıçta kaynak dosya boyutu ve SHA doğrulanmalı.

### 8.4 GPU provisioning otomatik olmalı

- Öncelik politikası: mevcut GPU → Colab GPU → RunPod/başka provider.
- Colab quota hatası 60 saniye içinde provider fallback tetiklemeli.
- CPU fallback audio-review/alignment için default olarak yasak olmalı.
- RunPod API ile provisioning; tarayıcı UI kullanılmamalı.
- Image/container digest sabitlenmeli.
- Model/cache network volume'da kalmalı.
- Bütçe üst sınırı, idle timeout ve hard deadline olmalı.

### 8.5 Transfer ve Drive doğrulaması

- Jupyter upload/download production transport değildir.
- Drive API veya rclone kullanılmalı.
- Upload: `.partial` → remote size/hash readback → exact name atomic rename.
- Path görünmesi sync kanıtı sayılmamalı.
- Final hash eşleşmeden stage PASS olamaz.

### 8.6 Audio-review veri modeli

Her karar şu alanlara sahip olmalı:

- `utterance_uid`
- `evidence_audio_sha256`
- `clip_start_ms`, `clip_end_ms`
- `target_start_ms`, `target_end_ms`
- `speech_start_ms`, `speech_end_ms`
- `decision`
- `reviewed_text_sha256`
- `model/version/device`
- `confidence`
- `review_timestamp`

Context padding hiçbir zaman speech coverage olarak değerlendirilmemeli.

### 8.7 Speaker-aware alignment

- Global non-overlap yasağı kaldırılmalı.
- Monotonicity aynı UID/speaker lane içinde zorunlu olmalı.
- Cross-speaker overlap geçerli bir durum olmalı.
- Bir utterance'ın kelimesi başka speaker'ın intervaline çekilememeli.
- Cue görünür başlangıç/bitişi ilk/son doğrulanmış kelimeye bağlanmalı; padding render edilmemeli.

### 8.8 Text ve timing mutation sınırı

- `TRCorrectedRecord` ayrı immutable artifact.
- `TimingOverride` yalnız zaman alanlarını değiştirebilir.
- `TextOverride` yalnız ayrı review stage'de ve audit hash'iyle uygulanabilir.
- Notebook runtime monkeypatch yasaklanmalı.
- Override dosyaları repoda version-controlled YAML/JSON ve fixture testli olmalı.

### 8.9 Gözlemlenebilirlik

Kullanıcının her an göreceği tek status objesi:

```json
{
  "episode": 12,
  "run_id": "...",
  "stage": "AUDIO_REVIEW",
  "status": "RUNNING",
  "processed": 10,
  "total": 352,
  "percent": 2.84,
  "elapsed_sec": 123,
  "eta_sec": 4200,
  "provider": "runpod",
  "gpu": "RTX 4090",
  "cost_so_far_usd": 0.42,
  "last_checkpoint_at": "...",
  "last_uid": "..."
}
```

Hata çıktısı mutlaka exact stage, UID, traceback, retryable/non-retryable sınıfı ve önerilen otomatik recovery action'ı içermeli.

### 8.10 Shutdown garantisi

- `try/finally` içinde remote upload verification sonrası stop.
- FAIL durumunda önce checkpoint/diagnostic bundle yükle, sonra stop.
- Heartbeat kaybolursa provider-side idle timeout.
- Maksimum bütçe veya süre aşımında hard stop.
- Stop sonrası provider state `NOT_RUNNING` ve `$0/hr` readback yapılmalı.

## 9. Astra için acceptance test listesi

Pipeline “hazır” sayılmadan şunlar otomatik test edilmelidir:

1. Colab GPU quota fail → 60 saniye içinde RunPod fallback.
2. Drive mount path var ama remote sync yok → upload verification fail.
3. Process kill → aynı completed UID set'inden resume.
4. Input ZIP/hash değişir → eski checkpoint reddedilir, source silinmez.
5. İki speaker aynı anda konuşur → ayrı cue'lar PASS, text merge olmaz.
6. Reviewed clip padding içerir → padding coverage gerektirmez.
7. Duplicate suffix/sessizlik → explicit non-dialogue evidence ile PASS.
8. Timing override text değiştirmeye çalışır → type/schema error.
9. `Ha?`, `Ne?`, `Ya.`, `Evet.` ile uzun metin ezilmesi → semantic shrink alarmı.
10. Block count/UID/order/timing değişir → ID translation validation hard fail.
11. 42 char/line, max 2 line, CPS limiti ve cue duration QA.
12. Source video ile final cue start/end playback spot checks.
13. Upload sonrası remote byte count + SHA mismatch → final marker yazılmaz.
14. Success/fail sonrası Pod auto-stop ve `$0/hr` doğrulaması.
15. Bölüm 12'deki 15 unresolved interval regression fixture'ı.

## 10. Önerilen hedef süreler

RTX 4090 veya benzer GPU, sıcak model cache ve otomatik Drive API ile:

| Stage | Hedef |
|---|---:|
| Source doğrulama + audio extract | 2–5 dk |
| ASR/VAD | 10–25 dk |
| TR text correction | 15–40 dk |
| Audio review | 10–25 dk |
| Forced alignment + coverage QA | 5–15 dk |
| ID translation | 10–30 dk |
| Finalize + upload + readback | 3–7 dk |
| Toplam sıcak-cache hedefi | yaklaşık 60–120 dk |

İlk 40 dakika fast-lane hedefi ayrıca **30–45 dakika** olmalı; fakat fast-lane strict run'ı bozmamalıdır.

## 11. Astra'ya verilecek kısa uygulama talimatı

> Bölüm 12 incident report'unu regression baseline olarak kullan. Notebook monkeypatch'lerini production tasarımı kabul etme. Pipeline'ı tek komutlu, state-machine tabanlı ve provider-independent kur. Source immutable olsun; her stage exact input/output hash, UID set ve config ile checkpointlensin. GPU yoksa CPU'ya sessiz düşme; provider fallback yap. Cross-speaker overlap'i geçerli kabul et fakat speaker metinlerini asla birleştirme. Review clip padding'i speech coverage sayma. Timing override'ın text değiştirmesini type seviyesinde engelle. Drive upload'ı remote byte/hash readback olmadan PASS sayma. Her sonuçta diagnostic/checkpoint'i kalıcılaştırıp RunPod'u `finally` içinde kapat. Strict V2 release ile emergency SRT yolunu klasör, marker ve dosya adlarında tamamen ayır. Bölüm 12'deki 15 unresolved interval ve uzun cümlenin `Ha?/Ne?/Ya./Evet.` ile ezilmesi vakalarını zorunlu regression test yap.

## 12. Son karar

Gecikmenin ana sebebi Whisper veya Endonezce çeviri hızı değildi. Son çeviri batch'i yalnız **15,3 saniye** sürdü. Asıl maliyet:

1. GPU provisioning'in önceden çözülmemesi,
2. stage/hash/checkpoint zincirinin kırılgan olması,
3. overlap ve review-padding validator tasarım hataları,
4. stale notebook override'ları,
5. Drive/RunPod transferinin production API yerine UI/Jupyter üzerinden yürütülmesi,
6. otomatik watchdog ve shutdown olmamasıydı.

Yeni pipeline bu altı noktayı çözmeden “tek tuş” deneyimi sağlamış sayılmamalıdır.


