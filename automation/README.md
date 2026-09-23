# Bölüm → Drive → Endonezce altyazısı gömülü MP4

Kullanıcının 23 Eylül son talebi bu akışın otoritesidir: **resmî altyazıyı bekleme**.
RunPod, açık PC, ücretli LLM API veya alternatif çeviri modeli kullanılmaz.

## Çalışan kod yolu

1. `official_subtitles.media(episode)` resmî tam bölüm sayfasından gerçek MP4 URL'sini okur.
   Fragman veya tahmin edilen CDN dosyası kullanılmaz.
2. `video_flow.prepare(root, episode, config)` önce videoyu `source.mp4` olarak
   Drive'a kaydeder; SHA-256 değişmezdir. Show TV'nin web oynatıcı bölge uyarısından
   doğrudan medya URL'sinin bulut ortamında da kapalı olduğu sonucu çıkarılmaz.
3. Aynı kaynağın Türkçe VTT'si hazırsa, süre ve kaynak eşleşmesiyle kullanılır.
   Yoksa veya kullanılamıyorsa **hemen Colab GPU large-v3 ASR** çalışır.
   `force_asr=True` gerçek yedek yol denemesi içindir. Başlanan yol checkpoint'e
   yazılır; ertesi gün çıkan VTT tamamlanmış ASR işini değiştirmez.
4. ChatGPT paketteki özgün `TRANSLATION_INSTRUCTIONS.md`, ad ve dinî terim
   sözlükleriyle çevirir. Görev kullanıcıdan ZIP taşımayı istemez; Drive üzerinden
   batch'leri okuyup tam doğrulanmış dönüş ZIP'ini kendisi oluşturur.
5. `video_flow.finish(root, returned_zip)` mevcut `engine/burned_mp4.py` ile
   `h264_nvenc` kullanarak MP4'e altyazıyı gömer. Eski Arial stili korunur.
   Kaynak süre, çözünürlük, H.264/AAC akışları, SRT/kaynak/çıktı hashleri kontrol edilir.
6. Gerçek dinleme bitmediyse çıktı `.draft.mp4` kalır. Test başarısı veya yayıncı
   zamanlarının değişmemesi ses senkronunun kanıtı değildir. SRT tek başına teslim değildir.

Bir kaynak ülke/erişim kısıtı verirse erişilebilir resmî alternatif bölüm URL'si
veya Drive kaynak videosu kullanılabilir. Otomatik proxy, bot engeli aşma döngüsü,
yapay keepalive veya kota hilesi yoktur. Sessiz CPU ASR fallback yoktur.

## Cuma görevi ve gerçek sınır

ChatGPT kontrol görevi **25 Eylül 2026 Cuma 06:00 Asia/Singapore** başlangıçlıdır;
saatte bir, en fazla 96 kontrol. Son talebe göre video/ASR/burn teslimini ister.
Bu kayıt tek başına Colab GPU veya Drive yetkisinin gelecekte hazır olacağını
kanıtlamaz. Görev ilk çalışmada erişimi kontrol eder; somut engeli bildirir.
Ücretsiz Colab oturumu, GPU kotası ve yeni Drive yetkisi kullanıcı adımı isteyebilir.

Repo: `ismailkorkmaz1905/ma-sub`, PR #2, `codex/colab-subtitles-no-forced-alignment`.
Drive hesabı: `ismailkorkmaz490@gmail.com`.
Proje klasörü: `1Gdn4WLjICGJNsYIMCSpSNyXgA_8mdL5p`.
Görev durumu: `Official_Subtitles/ep15-status.json` (önceki kimliği korunur).
Colab bölüm klasörü: `Muhtemel_Ask_Subtitles/Colab_v1/Muhtemel Ask 15.Bolum`.

## 23 Eylül canlı deneme

- Kullanıcının Colab hesabında gerçek Tesla T4 / 15 GB GPU tahsis edildi.
- Show TV 14. bölümün sayfada verilen MP4 URL'si Colab'dan HTTP 206 döndürdü:
  toplam kaynak boyutu 6,226,057,304 byte. Singapur web oynatıcı uyarısı bu erişimi engellemedi.
- Gerçek videonun 65–155 saniye aralığı indirildi; `force_asr=True` çalıştırıldı.
- İlk gerçek GPU denemesi `preprocessor_config.json` eksikliği nedeniyle 80/128 mel
  hatası verdi. Model indirme listesi düzeltildi; gerçek GPU'da yeniden çalıştı.
- Sessizlikleri birleştiren `vad_filter=True` denemesi erken kelime zamanları
  üretti ve reddedildi. Son yöntem geniş ASR bağlamı + ayrı konuşma aralığı ASR.
- Son akış 15 cue üretti; ChatGPT çevirisi eski NVIDIA burner ile gerçek MP4'e
  gömüldü: 1280×720, yaklaşık 90 saniye. İlk encode 14.62 saniye sürdü.
- Kaynak ve MP4 Drive'a yüklendi. Son MP4 bağımsız olarak Drive'dan geri indirilip
  boyut/SHA-256 eşleşmesi doğrulandı. Bunlar **90 saniyelik deneme**; tam bölüm değildir.
- Yazıların tek/iki satır görünümü gerçek karelerden kontrol edildi. Dinleme yapılmış
  gibi kayıt yazılmadı. Bağırma ve hızlı dua sahnesi çözülememiş ASR/senkron riskidir.
- Örnek çeviride birkaç belirsiz söz için yayıncı metni referans kullanıldı; bu,
  altyazısız yedeğin kusursuz çeviri yaptığını kanıtlamaz. Dosya DRAFT kalır.
- Tam bölüm kalitesi, gözetimsiz başlangıç ve bağımsız Drive readback ayrı konulardır;
  bu denemeyle tümü tamamlanmış sayılmaz.

## Devam eden görev için dosyalar

`source.mp4`, `source.json`, `video_workflow.json`, `schema.json`, `manifest.json`,
`asr/chunk_*.json`, `handoff/*PACK.zip`, `handoff/*TRANSLATED.zip`, `review.json`,
`video_output.json` ve `output/` korunur. Yayıncı yolu `captions/` altında çalışır.
Büyük video ve WAV'ı sohbet içine veya Git'e koyma. Küçük checkpoint ZIP'i güvenli
bağıl yollarla aç; dosya kimliği/hashleri uyuşmazsa mevcut çıktıları ezme.

Drive'a yerel bağlı dosya sistemi üzerinden hash kontrolü, bağımsız Drive API
readback değildir. Büyük dosya okumada connector'ın 256 MiB sınırını hesaba kat;
tam MP4 uzak byte/hash doğrulaması yapılmadan COMPLETE durumu yazma.

Gerçek çalışma makbuzu: [live-test-2026-09-23.json](live-test-2026-09-23.json).
Drive notebook: `1ZsUB_JFQcQnlEONhf9ry1JV5SeBIUIpJ`.

## Bekleyen gerçek izin

Colab `Mount Drive` girişimi otomatik onay kontrolünde reddedildi: notebook'un
yalnız proje klasörü yerine tüm Drive'a geniş erişim istemesi gerekçe gösterildi.
Bu yetki verilmedi; başka yoldan aynı geniş erişim açılmadı. Mevcut Drive
connector'ı ile 90 saniyelik kaynak/MP4 teslimi yapılabildi. Tam bölümün Colab'dan
Drive'a doğrudan checkpoint/kayıt yolu için kullanıcıdan bu özel izin istenmelidir.
Görev izin yokken hazırmış gibi davranmamalı veya resmî altyazıyı beklemeye dönmemeli.
