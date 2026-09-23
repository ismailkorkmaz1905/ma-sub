# İnceleme notu — 23 Eylül 2026

## Yapılan değişiklik

`video_flow.py` mevcut video gömme kodunu yeni Colab girişine bağlar. Akış:
video indir → varsa aynı kaynağın Türkçe altyazısı, yoksa GPU ASR → mevcut
ChatGPT ile çeviri → NVIDIA ile altyazısı gömülü MP4 → Drive doğrulaması.
Resmî altyazı beklenmez. RunPod ve ücretli çeviri API'si yoktur.

## Gerçek denemede bulunanlar

- Kullanıcının Colab hesabında Tesla T4 tahsis edildi.
- Singapur web oynatıcı uyarısına rağmen sayfadaki gerçek Show TV MP4 URL'sine
  Colab'dan erişildi. 14. bölümün 65–155 saniyesi gerçek video olarak indirildi.
- İlk GPU koşusu, eksik `preprocessor_config.json` nedeniyle 80/128 mel hatası verdi.
  Dosya indirme listesine eklendi; gerçek model tekrar çalıştırıldı.
- Geniş Whisper kelime zamanları sessizlikte erkene kayabiliyordu. Sessizlikleri
  birleştiren `vad_filter=True` denemesi de reddedildi. Son yöntem, geniş ASR'yi
  metin bağlamı olarak koruyup ayrı konuşma aralıklarında yeniden ASR çalıştırır.
  Düzeltilmiş metne CTC/WhisperX forced alignment uygulanmaz.
- Aynı model iki kez kullanılır; bağımsız akustik doğrulayıcı diye sunulmaz.
- Gerçek 90 saniyelik videoya 15 Endonezce cue gömüldü. Eski Arial stili, H.264
  NVIDIA kodlayıcı ve Türkçe AAC ses kullanıldı. Son encode yaklaşık 14.3 saniye.
- Dosya boyutu/hash/akış/süre kontrolü yapıldı. Gerçek karelerde bir ve iki satırlı
  altyazıların görüntüye gömüldüğü görüldü. Drive kopyası geri indirilip hash eşleşti.

## Kalan sınırlar

- Bu tam bölüm testi değildir. Dinleme yapılmış gibi kayıt yazılmadı.
- Bağırılan isim, hızlı dua, kısa cue ve olası eksik konuşma yerleri inceleme ister.
  Örnek çeviride birkaç belirsizlik yayıncı metniyle düzeltildi. ASR yedeğinin
  tek başına kusursuz anlam/senkron sağladığı iddia edilmez. Çıktı DRAFT'tır.
- Cuma 25 Eylül 06:00 Asia/Singapore kontrol görevi kayıtlıdır. Colab Pro/L4 ve
  kullanıcı onayı sonrası Drive bağlama artık canlı doğrulandı. Gelecekteki GPU
  tahsisi yine garanti değildir. Güncel L4/kota/1080p kanıtı automation/README.md içindedir.

## Kanıt ve denetim

[`automation/live-test-2026-09-23.json`](../automation/live-test-2026-09-23.json)
cihazı, kaynak aralığını, hashleri ve gerçek teslim dosyalarını kaydeder.
Hedefli sözleşme testleri, tam mevcut test paketi ve `git diff --check` çalıştırıldı.
Yazılım testleri gerçek videodaki ses/çeviri kalite sorunlarını geçersiz kılmaz.

Opus incelemesinde özellikle: konuşma dışı geniş-ASR adayları, kısa seslerde isim
hataları, işaretlenmiş zamanlar, parça sınırı uyarıları, checkpoint devamı ve
Drive izin kapsamı kontrol edilmelidir. Eski üretim/RunPod/forced-align kodu silinmedi.

## Son L4 doğrulaması

Üç 1080p parçada gerçek ASR, 60 saniyelik bir parçada doğal Endonezce çeviri ve
7.216 saniyede GPU yakma tamamlandı. 5 GB sınırı ölçeklenmiş gerçek boyutla
denendi; tam bölüm yapılmadı. Notebook'un kendi kapanış hücresi Drive'ı flush
edip L4'ü serbest bıraktı. 80 başlangıç bakiyesi 79.16 oldu.

Sahte stok sözler ham kanıtları korunarak açık incelemeye ayrılıyor. Zamanı
erkene çeken bağlam birleştirme denemesi reddedildi. Son örnek 26 açık QA maddesi
ile DRAFT: bunlar 26 ayrı işitilmiş hata anlamına gelmez; dinleme gereği, kısa
süreler ve belirsizlik işaretlerini içerir. Eyvallah satırında mevcut Allah
kontrolünün alt-dize eşleşmesi ayrıca bir uyarı üretiyor; kural gevşetilmedi.
