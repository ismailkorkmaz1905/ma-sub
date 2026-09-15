# Dort Saatlik Uretim Plani - 2026-09-12

14 Eylul uygulama notu: Limit reseti sonrasi kullanici kod ve yerel testleri yeniden yetkilendirdi. Ilk uygulama paketi ve kalan isler `docs/PIPELINE_IMPLEMENTATION_2026-09-14.md` dosyasindadir. Asagidaki "bu tur kod yazilmadi" ve "uygulanmamistir" ifadeleri 12 Eylul planlama kesimine aittir; guncel uygulama durumu degildir. Ucretli RunPod, canli yayin ve gercek bolum provasi halen bu uygulama turunun disindadir.

Durum: Yalniz mimari plan. Bu tur kod yazilmadi. Uygulama, kaynak kodu ve test degisiklikleri limit sifirlanana kadar durduruldu. Plan mevcut `main` HEAD `c454a90` uzerine kuruludur. Bu commit monitor durumunu, logu ve checkpoint ozetini tek bounded `poll` isteginde toplar.

## Hedef ve sert sinirlar

- Hedef: Kaynak hazirligindan Drive SHA-256 readback sonucuna kadar 240 dakika. Bu bir muhendislik hedefidir, garanti degildir.
- Mutlak ust sinir: 360 dakika. Bu sure sonunda eksik is `BLOCKED` olur; dusuk kaliteli veya eksik bir sonuc `PASS` yapilmaz.
- Ornek 06:20 baslangicinda hedef bitis 10:20, mutlak kesim 12:20'dir. Dort saatlik hedef kacirilinca kalite esigi dusurulmez; kalan iki saat yalniz tanimli rezervdir.
- Insan duzeltmesi, ceviri beklemesi, yeniden deneme ve yayin sureleri ayni duvar saatine dahildir. Asama degisince saat sifirlanmaz.
- GPU yalniz GPU gerektiren ASR ve akustik kanit islerinde tutulur. Ceviri, yerel encode ve Drive yayininda ucretli Pod serbest birakilir.
- Kaynak medya, hashler, duzeltilmis Turkce metin, UID/sira, bilinen konusmaci ayrimi, Drive byte/SHA readback ve strict/emergency artifact ayrimi zayiflatilmaz.
- Mevcut kod 12 asama bildirimi gonderir. Asagidaki 11 asama, `strict_finalize` ile encode hazirligini tek sonuc kapisinda gruplandirilan kritik yol modelidir; bildirim adlari veya artifact sozlesmeleri sessizce degistirilmez.

## Olculmus taban

| Bulgu | Kaynak | Sonuc |
|---|---|---|
| 92 CLI logu; 89 tamamlanmis, 3 bitisi UNKNOWN; kesim penceresi 30 saat 45 dakika 3 saniye | `docs/EP13_TIMELINE_AUDIT_2026-09-12.md` | Tum takvim suresini GPU veya bos bekleme diye siniflandirmak desteklenmiyor. |
| Forced alignment 28 benzersiz baslama; 23 FAIL ve 5 terminal UNKNOWN | Ayni audit | 23 FAIL toplam 6 saat 19 dakika 25.7 saniye; 5 UNKNOWN son heartbeatleri en az 1 saat 21 dakika 7.5 saniye daha gosterir. Bu GPU fatura suresi degildir. |
| Benzersiz terminal asama toplam 8 saat 40 dakika 51.6 saniye | Ayni audit | Replay ayiklanmadan yapilan eski toplamlar kritik yol karari icin kullanilmaz. |
| 46 logda `10 upgraded, 126 newly installed` | Ayni audit | Tam bootstrap suresi UNKNOWN; immutable image baglama yine oncelikli altyapi isidir. |
| 521 `sent` log satiri, 482 benzersiz Message-ID | Ayni audit | 39 satir replay gorunumudur. SMTP toplam suresi UNKNOWN; mail azaltmanin maddi sure kazanci olculmeden iddia edilmez. |
| Gercek tam-bolum QSV encode receipt'i 1,940.25 saniye | Ayni audit | 32 dakika 20.25 saniyelik encode kanitidir; yeni production akisinin tamamini kanitlamaz. |

Episode 13 emergency segment-timing artifacti ile kaynak coarse zamanlari arasindaki yerel olcum 2,886 cue icindir: baslangic kaymasi 500 ms'den buyuk 404, 1,000 ms'den buyuk 184, 2,000 ms'den buyuk 38 cue; maksimum 3,680 ms ve p95 1,180 ms. Maksimum bitis kaymasi 3,920 ms'dir. Bu degerler akustik hata olcumu degil, yalniz source-coarse-relative drift olcumudur. Bu nedenle global forward-shift emergency geometrisi gelecekteki strict kalite tabani veya akustik dogruluk kaniti olamaz.

Ayni artifactte 7 saniyeden uzun 21 cue ve 21,390 ms maksimum sure vardir. 42 karakteri asan 139 satir ve 78 karakter maksimum satir uzunlugu bulunur. `docs/EP12_ACCEPTANCE.md` 42 karakteri hard kabul kosulu sayarken `config/production/series.yaml` hard `qa_max_chars_per_line=84`, target 42 kullanir; `wrap_text` de satir basina 84 hard limit uygular. Bu bir politika celiskisidir, mevcut production sonucunun kesin 42-gate FAIL oldugu iddiasi degildir. Ceviri pack'i dondurulmadan once tek otorite secilmelidir: hard 42 ya da gerekcesi, gorunur kalite kaniti ve onayi kayitli bilincli istisna.

## 11 asamali kritik yol

| No | Sonuc asamasi ve mevcut kod temeli | Bagimlilik | Hedef pencere | Sert kilometre tasi | Kabul kriteri |
|---:|---|---|---:|---:|---|
| 1 | Download: `pipeline._resolve_source_url`, `download_source`, `_guard_existing_source` | Preflight | T+0 - 8 dk | T+30 dk | Kaynak immutable SHA-256, yeterli yerel/Drive alan, kimlikler, kota ve credential preflight PASS. |
| 2 | Audio: `extract_audio` | 1 | T+8 - 15 dk | T+30 dk | Ses artifact hashli, sure ve teknik probe kaynakla tutarli. |
| 3 | Raw ASR: `transcribe_raw_audio`, `RawASRConfig` | 2, immutable image | T+15 - 45 dk | T+90 dk | GPU-only calisma, model/runtime kimligi, word/VAD/coarse provenance ve checkpoint hashleri dogrulanmis. CPU fallback yok. |
| 4 | TR pack: `create_tr_correction_pack` | 3 | T+45 - 50 dk | T+100 dk | UID/sira/hash bagli pack eksiksiz ve yeniden uretilebilir. |
| 5 | TR return: `validate_tr_correction_output` | 4 | T+50 - 80 dk | T+150 dk | Otomatik, guvenilir duzeltme akisinda blank/ekleme/isim-sayi kaybi yok; belirsiz replik akustik kanitla cozulur. Cozulemeyen kayit bounded `BLOCKED` olur, insan beklemez. |
| 6 | Audio review: `resolve_tr_audio_reviews` | 5 | T+80 - 100 dk | T+180 dk | Hash-bound akustik kanitla deterministic karar; unknown veya yetersiz kanit fail-closed `BLOCKED`. |
| 7 | Forced alignment: `pipeline.align`, `_aligned_checkpoint`, `correction_records_to_alignment_inputs` | 6 | T+100 - 135 dk | T+225 dk | Mevcut strict CTC kontrati korunur; window checkpoint/cache kullanilir; overlap, drift, VAD, duration ve global validation PASS. |
| 8 | ID pack: `build_strict_artifacts`, `create_id_translation_pack` | 7 | T+135 - 140 dk | T+235 dk | Timing/readability politikasi kesinlesmis, exact translation pack dondurulmus; freeze sonrasi UID/sira/timing degismez. |
| 9 | ID return: `load_and_validate_id_translation_zip` | 8 | T+140 - 155 dk | T+265 dk | Paralel batch translation ve otomatik semantic/register/name/number/religious/readability QA PASS; cozulmeyen cue `BLOCKED`. |
| 10 | Finalize + encode: `finalize_episode`, `plan_encoding_settings`, `create_encoding_samples`, `burn_indonesian_mp4` | 9, yerel QSV onayi | T+155 - 215 dk | T+325 dk | Strict hash zinciri ve subtitle QA PASS; tam encode sure/stream/subtitle gorunurlugu/output SHA-256 dogrulanmis. |
| 11 | Drive readback: `publish_local_delivery`, `remote.upload_verified` | 10, Pod release | T+215 - 240 dk | T+360 dk | Collision inventory korunmus, final remote byte sayisi ve SHA-256 yerelle eslesmis; ancak bundan sonra delivery PASS. |

Hedef pencereler toplam 240 dakikadir. TR dondurulunca ID ceviri taslagi Stage 6-7 ile paralel hazirlanabilir, fakat exact ID pack Stage 8'den once otorite olmaz. T+240 sonrasi kalan 120 dakika yalniz olculmus sapma rezervidir; yeni ozellik, sinirsiz retry veya ayni hatanin yeniden calistirilmasi icin kullanilmaz. Runtime yolu zorunlu insan yaniti beklemez: otomatik kanit karar veremiyorsa bounded `BLOCKED` olur. T+360 `BLOCKED`, tamamlanmis delivery vaadi degil; kaliteyi koruyan terminal guvenlik sonucudur.

Bu pencereler olculmus yeni-run sureleri degil, qualification icin tasarim butceleridir. Drive hedefi 25 dakika, sert rezervi en az 35 dakikadir; gercek HQ upload + tam readback olcumu bu butceyi asarsa plan yeniden dengelenmeden 4 saatlik teslim iddiasi yapilmaz. Calisma sirasinda kalan sure, geriye kalan islerin olculmus asgari butcesinin altina dustugunde rezerv ihlali erken bildirilir; kalite kontrolleri atlanmaz.

Sonradan gelen gercek HQ teslim olcumu 54 dakika 5 saniyedir (12:42:59 - 13:37:04 SGT); mevcut 25/35 dakikalik Drive butcesini gecmistir. Iki yeniden aktarim ve ayni donemde ortak istemci API kota hatalari gozlenmistir. Dolayisiyla Drive kota/bounded-resume duzeltmesi ve yeniden olcum, 4 saat iddiasinin on kosuludur; bugunku teslim bu hedefi kanitlamaz. Ayrinti yeni timeline audit'indeki HQ teslim ekindedir.

## Oncelikli is paketleri

### P0 - Sure butcesi ve degismeyen hata korumasi

Dosyalar: `config/runtime_policy.json`, `src/mas/pipeline.py`, `src/mas/runpod_controller.py`; testler `tests/test_pipeline_runtime.py`, `tests/test_controller_job_flow.py`, `tests/test_runpod_controller.py`.

- `target_episode_seconds=14400` gozlemsel ayar olmaktan cikarilip stage ve episode wall-clock kapilarina baglanir.
- Failure invariant; job/input kimligi, stage, code-dependency-model kimligi, hata tipi ve diagnostic artifact hashlerinden olusur.
- Ayni invariant yeniden gorulurse yeni ucretli Pod acilmaz ve sonuc `BLOCKED` olur. Degisen kanit yalniz bounded retry hakkini acar.
- Kod veya parametre hashini degistirmek tek basina yeni tam GPU denemesini hakli cikarmaz. Once retained hata kanitindan uretilmis yerel fixture ilgili bug icin PASS olmali; sonra yalniz etkilenen component icin resume plani cikmali ve ayni global episode butcesine sigmalidir. Her kucuk yamayi yeni tam-bolum ucretli provaya ceviren gelistirme dongusu normal haftalik runtime'in parcasi olamaz.
- Bu guard yalniz tekrar masrafini ve zamani keser; 4 saat icinde dogru subtitle/delivery uretimini tek basina saglamaz. Dogru sonuc icin residual kok-neden paketi de zorunludur.
- Kabul: Unit test ayni hatada relaunch sayisini 0, degisen artifact hashinde bounded retry'yi ve T+360'ta fail-closed `BLOCKED` sonucunu kanitlar.

### P0 - Immutable bootstrap image ve apt fallback

Dosyalar: `Dockerfile`, `runpod/bootstrap.sh`, `src/mas/runpod_controller.py`; testler `tests/test_runpod_controller.py`, `tests/test_controller_job_flow.py`.

- Repoda gercek bir image digest yoktur. Digest uretilmez veya tahmin edilmez.
- `Dockerfile` bagimliliklari `/opt/venv` altina kurar; controller ise `MAS_VENV_DIR=/workspace/ma-sub/.venv` gonderir ve remote job'u bu ikinci Python yolundan baslatir. Ayrica yeni Pod icin protected Pod'un `imageName` degerini kopyalar. Dolayisiyla yalniz image build etmek yetmez: gercek calistirilan image, bootstrap ortam yolu ve remote job Python yolu birlikte ayni dogrulanmis runtime'a baglanmalidir.
- Provider tarafinda gozlenmis image digest config'e girilmeden production Pod olusturma fail-closed olur; controller istenen digest ile provider'in calistirdigi image kimligini esitlik testiyle dogrular.
- Mevcut runtime ABI marker; requirements SHA, Python ABI, distro/glibc, paketler, Torch/CUDA/cuDNN/C++ ABI ve ffmpeg/ffprobe yol-surum-SHA bilgisini korur.
- `bootstrap.sh` apt fallback'i recovery icin kalir, ancak 4 saat SLO'sunun normal yolu sayilmaz.
- Kabul: Temiz Pod'da gozlenen 126 yeni paket ve 10 guncelleme normal yolda yok; image/digest uyusmazligi is baslamadan FAIL; marker uyusmazligi cache reuse etmez.

### P0 - Mevcut strict CTC'yi hizlandirma

Dosyalar: `src/mas/pipeline.py`, `src/mas/engine/forced_align.py`; testler `tests/engine/test_forced_align.py`, `tests/engine/test_timing_qa.py`, `tests/engine/test_finalize.py`, `tests/test_pipeline_runtime.py`.

- Varsayilan production yolu mevcut CTC kontratini, `forced_alignment_v2.done.json`, aligned checkpoint, `strict_inputs` ve `finalize_episode` zorunlulugunu korur.
- `UnitJournal` basarili raw CTC cagrilarini zaten audio/model state/WhisperX/dependency/producer kimlikleriyle cache eder. Yeni bir raw CTC cache varmis gibi tasarlanmaz.
- Eksik olan, degismeyen raw cache sonucunun her run yeniden normalize/candidate-select edilmesini onleyen hash-bound per-conflict-component postprocess checkpoint'tir. Invalidation girdileri raw UnitJournal sonucu, ilgili cue/TR/speaker/coarse bounds, resolver surumu, dependency/model kimligi ve candidate/validation politika hashidir.
- `_resolve_alignment_overlaps` icindeki `attempted_joint_calls` yalniz in-memory'dir. Radius 1, 2, 4, 8 recovery ve global candidate kombinasyonlari zaten finite bounded'dir; ayni component evidence degismediyse checkpoint yeniden kullanilir.
- Deterministik ayni FAIL invariant'i yeni Pod acmadan `BLOCKED` yapar. Mevcut max-candidate, score, VAD, duration, drift ve final global overlap esikleri sure kazanmak icin gevsetilmez.
- Kabul: Ilk kosu ile ayni artifact hashlerini veren cache-hit kosusunda raw CTC ve component postprocess yeniden calismaz; tek cue/evidence/policy hash degisikligi yalniz ilgili component'i invalidate eder; output byte/hash ve strict validator sonucu cold run ile aynidir.

### P0 - Forced-alignment residual kok nedeni

Dosyalar: `src/mas/engine/forced_align.py`; testler `tests/engine/test_forced_align.py`.

Latest tamamlanmis alignment diagnostic'i 36 benzersiz residual cift gostermistir: 25 disjoint/touch, 11 coarse-overlap; 11 overlap'in tamami `speech_hole_rescue_asr` icerir. Bu geometri timeline toplamindan ayri artifact-level kanittir.

- Her denemeden once live residual seti yeniden hesaplanir; stale pair uzerinde calisilmaz.
- Gap/touch ordered pair icin tek-tarafli acoustic alignment giris pencereleri gercek `L.coarse_end` ve `R.coarse_start` kenarlarini kullanir. Sonuc zamanlari bu kenarlara zorla kesilmez; yeniden akustik hizalamadan elde edilir.
- Coarse-overlap icin conservative exclusion adayi, sol giris penceresini `R.coarse_start` noktasinda bitirip sag giris penceresini `L.coarse_end` noktasinda baslatmayi dener. Bu yalniz test edilecek pencere hipotezidir, dogrulanmis cozum veya nihai kelime zamanlarina trim degildir. Bilinen farkli konusmaci overlap politikasi korunur; metin birlestirilmez.
- Multi-cue residual component icin component-local aday uretilir; orta cue iki incident cut alabilir, ilk/son cue dis item context'ini korur.
- Her aday mevcut score, VAD, duration, drift ve final global overlap validation'dan gecer. UID/Episode allowlist, evrensel forward-shift, post-hoc timestamp trim veya esik gevsetme yoktur.
- Rescue-hole overlap'i speaker ve timed acoustic evidence ile genel olarak cozulur. Same-known speaker non-overlap zorunludur; different-known speaker overlap korunabilir; unknown yetersiz kanitta `BLOCKED` olur.
- Kabul: Gap, touch, overlap-exclusion, 3-cue ortanin iki cut almasi, speaker sinirlari, stale-residual recompute ve final global rejection fixture'lari gecer. Ilk production kapisi dogru strict sonuc ve hash-bound bounded resume'u birlikte kanitlar.

### Kosullu arastirma - CTC kontrat migrasyonu

Tam CTC'yi delivery kritik yolundan cikarmak bir hiz ayari degildir. Mevcut 23 FAIL'in toplam suresi, tek bir basarili strict calismanin 4 saate sigamayacagini kanitlamaz; asil kanit, tekrarlanan calismalarin toplam maliyetidir. CTC kanitsiz opsiyonel yapilmaz ve default yol degistirilmez. Ancak strict optimizasyonlardan sonra ayri bir kontrat degisikligi kullanici tarafindan onaylanir ve akustik kalite esdegerligi karsilastirmali olarak kanitlanirsa su arastirma yapilabilir:

1. Yeni schema/marker/receipt surumu eklenir; eski strict consumer aynen kalir.
2. Segment timing kaniti coarse cue bounds, raw word/VAD provenance, speaker politikasi ve hedefli acoustic review kararlarini hash ile baglar.
3. Bilinen farkli konusmaci overlap'i korunur ve metinler asla birlestirilmez. Ayni konusmaci non-overlap olmak zorundadir. Bir veya iki unknown konusmaci review evidence uretir.
4. Mevcut strict CTC ile ayni bolumler uzerinde boundary, word coverage, speaker overlap, perceptual sync ve failure-recall karsilastirmasi yapilir. Esdegerlik kaniti olmadan CTC opsiyonel olmaz.
5. Emergency `forward-shift-with-readability-extension-v1`, strict artifact/marker olarak tasinmaz veya yeniden adlandirilmaz.

Kabul: Eski strict artifactler ayni validator ile gecer; yeni versioned artifact forged hash/provenance ile reddedilir; speaker overlap politikasi ve cue boundary testleri gecer; karsilastirmali akustik kalite strict CTC'ye esdegerligi gostermeden production default switch edilmez.

### P0 - Turkce ve ceviri kalite kapisi

Dosyalar: TR ve ID pack validatorlari, `src/mas/engine/id_translation.py`, timing/translation QA; testler `tests/engine/test_id_translation.py`, `tests/engine/test_translation_validation_regressions.py`, `tests/engine/test_timing_qa.py`.

- Mevcut Turkce kaynakta belirsiz replikler ceviri sezgisiyle degil, ilgili kaynak ses penceresindeki otomatik akustik kanitla cozulur. Kanit yetersizse runtime insan beklemez, `BLOCKED` olur.
- Duzeltilmis TR metin dondurulduktan sonra 3 baglamli ceviri batch'i paralel calisir. Tum cue'lara raw MT yapip sonra yogun toplu yeniden ceviri uygulanmaz; ayri risk-odakli QA isim, sayi, anlam, duygu, register, dini ifade ve belirsiz replik uydurmasini arar. Segmentleme ve satir/CPS politikasi kesinlesince exact translation pack freeze yapilir.
- Register sahne ve iliskiye gore dogal `aku/kamu/nggak/udah/aja` ya da resmi `Pak/Bu/Anda` olur; isimler ve sayilar korunur.
- Religion canon en az su eslesmeleri korur: `Allah askina` -> `Demi Allah`; `Allah'im` ve `Allah Allah` -> `Ya Allah`; `Insallah` -> `Insyaallah`; `Masallah` -> `Masyaallah`. Kaynakta Allah bulunan otoriter metin yalniz `Semoga` diye cevrilmez, kaynak Allah referansi korunur. `vallahi` ve `eyvallah` substring false-positive uretmez.
- Freeze sonrasi UID, order, timing ve source text degisirse pack hash'i gecersiz olur ve yeniden validation gerekir.
- 42/84 satir limiti celiskisi tek config/schema otoritesinde cozulmeden release yapilmaz.
- Kabul: Isim/sayi/dini ifade, anlam kaybi/ekleme, register, blank, duplicate UID, order, line length, CPS ve uzun cue testleri otomatik gecer. Riskli cue insan karsilastirmasi production runtime'inin bekleme noktasi degil, yeni yolun release qualification kanitidir.

### P1 - Diagnostic hash dedup

Dosyalar: `src/mas/runpod_controller.py`, gerekirse `src/mas/remote_job.py`; testler `tests/test_controller_job_flow.py`, `tests/test_remote_job.py`.

- Remote taraf bounded manifest verir: guvenli relative path, byte size, SHA-256.
- Canonical veya retained local dosya ayni size ve SHA'ya sahipse SCP yapilmaz.
- Mevcut 120 saniyelik shared grace, missing artifact warning, path safety, indirme sonrasi hash validation ve immutable `remote-checkpoints/<sha>/...` evidence semantigi korunur.
- Kabul: Manifestte ayni size/SHA ile eslesen retained evidence yeniden indirilmez; farkli hash indirilir ve dogrulanir; forged path/hash FAIL olur. Mevcut loglardan tekrar byte toplami ve kazanilacak sure UNKNOWN oldugu icin performans kazanci ayrica olculur.

### P1 - Yerel QSV encode yolu

Dosyalar: `src/mas/engine/burned_mp4.py`, pipeline/controller delivery ayrimi; testler `tests/engine/test_burned_mp4.py`, `tests/test_production_delivery_flow.py`.

- Yerel Intel QSV/ffmpeg capability probe ve driver/encoder kimligi receipt'e yazilir.
- Bas, orta, son ve zor sahne sample'lari teknik ve gorunur kalite kapisindan gecer.
- GPU ASR/akustik isinden sonra Pod kapatilir; encode yerelde baslar. Mevcut external-controller `h264_nvenc` zorunlulugu versioned execution plan ile acikca degistirilir.
- Kabul: Mevcut 1,940.25 saniyelik tam-bolum QSV encode receipt'ine ek olarak temsili sample seti ve tam bolum stream/sure/hash dogrulamasi vardir. QSV yoksa sessiz CPU fallback olmaz; acik NVENC plani veya `BLOCKED` olur.

### P1 - Drive alan, OAuth kota ve verified delivery

Dosyalar: `src/mas/delivery.py`, `src/mas/remote.py`; testler `tests/test_delivery.py`, `tests/test_production_delivery_flow.py`.

- Encode oncesi yerel alan, yayin oncesi remote alan ve kota preflight yapilir.
- Shared rclone istemcisi yerine kullaniciya ait private OAuth client kullanilir; quota/credential kimligi receipt'e baglanir.
- Resumable partial upload, exact filename collision inventory, onceki nesnenin byte/SHA kanitiyla korunmasi ve final remote byte/SHA readback aynen kalir.
- Kabul: Yetersiz alan veya kota encode/yayin oncesi fail-closed; upload resume eder; byte ya da SHA uyusmazligi delivery PASS vermez.

### P2 - Mail kritik yoldan cikarma

Dosyalar: `src/mas/notify.py`, pipeline stage event cagrilari; testler `tests/test_notify.py`.

- Synchronous SMTP her start/PASS olayinda calistirilmaz. Hashli local event spool tutulur; yalniz anlamli state transition, user-action, terminal PASS/FAIL gonderilir.
- Paid GPU lease kapandiktan sonra bounded SMTP drain yapilir. Mevcut kaynakta `send_email` senkrondur; 20 saniye timeout, en fazla 3 deneme ve permanent 5xx no-retry davranisi korunur.
- Mail hatasi subtitle sonucunu FAIL yapmaz; unsent event evidence korunur.
- Kabul: Stage basina SMTP beklemesi kritik yol hesabinda 0 saniye; terminal durum kaybolmaz; tekrar gonderimler idempotenttir.

## Kosullu CTC alternatifi icin gerekli kalite kaniti

Bu matris production default'unu bugun degistirmez. Tam CTC'nin delivery yolundan alinmasi ancak asagidaki otomatik kontroller ve ayri release qualification karsilastirmalari birlikte gecerken degerlendirilebilir:

| Kanit | Runtime otomatik kontrol | Release qualification kaniti |
|---|---|---|
| Metin otoritesi | TR/ID hash, UID/order, blank/duplicate, isim/sayi/dini ifade; cozulmeyen kayit `BLOCKED` | Belirsiz Turkce ve ID register/anlam icin temsili insan karsilastirmasi |
| Akustik provenance | Raw ASR timed words, VAD ve coarse-window hashleri; forged evidence rejection | Yuksek drift, rescue-hole ve unknown-speaker cue'larda strict CTC ile exact-window karsilastirmasi |
| Konusmaci | Known-different overlap korunur; same-speaker overlap FAIL; unknown `BLOCKED` | Unknown ve celiskili speaker fixture'larinda recall karsilastirmasi |
| Zaman geometrisi | Monotonic bounds, sure, CPS, satir, no forbidden overlap | Bas/orta/son ve conflict hotspot subtitle-video karsilastirmasi |
| Encode | Codec/stream/sure/frame probe, output SHA-256 | Temsili sample'larda okunabilirlik ve senkron algisi qualification'i |
| Yayin | Remote byte sayisi ve SHA-256 readback | Gerekmez; otomatik exact readback otoritedir |

Global forward-shift bu matriste yalniz geometrik non-overlap uretebilir; akustik provenance veya perceptual sync kaniti yerine gecmez.

## Bagimlilik ve rollout sirasi

Agent koordinasyonu: Koordinator ile birlikte en fazla 3 bagimsiz ceviri parcasi eszamanli calisir; her parcanin tek sahibi ve kesin UID kapsamli cikti kontrati olur. Baglam icin komsu replikler salt okunur verilir, cikti kapsamlarini genisletmez. Toplu yeniden ceviri yerine yalniz degisen veya QA'da isaretlenen kayitlar yeniden islenir. Uzun encode/transfer isleri gorunur ilerleme dosyasi, no-progress watchdog ve mutlak sure siniri yazar; ajan tek opak arac cagrisinda tum aktarim bitene kadar durum bilgisini tutmaz. Bu koordinasyonun gecmise ait token/zaman tasarrufu olculmus degildir.

1. P0 sure/unchanged-failure guard ve immutable image tamamlanir. Bunlar olmadan yeni performans kosusu baslatilmaz.
2. Mevcut strict CTC icin component postprocess checkpoint, invalidation ve unchanged-failure testleri tamamlanir; hicbir kalite esigi gevsetilmez.
3. 42/84 tek otorite karari ve ceviri pack freeze sirasi dondurulur.
4. QSV sample ve tam encode kaniti alinir; Drive alan/OAuth preflight ile verified upload staging'de gecer.
5. Tek gercek GPU bolum provasi 240 dakika hedef ve 360 dakika ust sinirle yapilir. Her kilometre tasi wall-clock ve artifact hashleriyle receipt'e yazilir.
6. Ancak gercek GPU run, tam encode ve canli Drive readback sonuclari PASS ise optimize strict yol production kabul edilir. Yerel test, CI, fixture, sample ve gercek GPU/Drive kanitlari raporda ayri etiketlenir.
7. CTC alternatifi bundan sonra ayri, kosullu arastirmadir. Akustik esdegerlik qualification'i ve acik sozlesme onayi olmadan default switch yapilmaz.

## Push, Drive ve mailin kritik yol durumu

- `src/mas/` ve `runpod/` production episode yolunda `git push` yoktur. Testli release bir push ile immutable episode/code SHA uzerinden calisir; her episode icin iki push gerektigi iddiasi yanlistir. `.github/workflows/ensure-v010-release.yml`, `release-v0.1.0.yml` ve `final-v010-gate.yml` eski self-mutating push/tag davranisi tasir, fakat yalniz `main` push ve kendi workflow path degisikliklerinde tetiklenir. Bunlari normal release yolundan ayirma P2 guvenlik/temizlik isidir, olculmus 30 saat 45 dakikanin nedeni degildir. Stable tag olusturma bu planin yetkisinde degildir.
- Drive upload ve final byte/SHA-256 readback Stage 11 olarak runtime kritik yolundadir. `src/mas/remote.py` icindeki verified upload yolu yerel SHA'yi alir, partial upload yapar, remote nesneyi tekrar okuyup byte/SHA esitligini denetler ve collision preservation receipt'i yazar. Bu readback olmadan delivery PASS yoktur.
- Mail cagrisi mevcut kaynakta synchronous oldugu icin calistigi noktada bekletebilir, ancak denetimde 482 benzersiz gonderimin toplam SMTP suresi UNKNOWN'dur. Mailin Episode 13 gecikmesindeki payi veya spool kazanci olculmeden sayisal hizlanma iddiasi yapilmaz. Mail subtitle/Drive sonuc kapisi degildir.

## Mevcut durum

- `c454a90` monitor polling optimizasyonu uygulanmis ve commitlidir.
- Diagnostic dedup, unchanged-failure guard, immutable digest binding, CTC postprocess checkpoint, kosullu CTC kontrat arastirmasi, QSV controller ayrimi, Drive preflight ve mail spool bu belgede planlanmistir; uygulanmamistir.
- Bu tur yalniz bu plan belgesi yazildi; source/test degisikligi yapilmadi.
- Reset sonrasi uygulama sirasi: sure/unchanged-failure guard, immutable digest binding, strict CTC component checkpoint, kalite/policy testleri, diagnostic dedup, QSV ayrimi, Drive preflight, mail spool, sonra tek bounded gercek GPU provasi.
- Limit sifirlanana kadar source/test degisikligi, test kosusu, yeni Pod, commit veya push yapilmayacaktir.
