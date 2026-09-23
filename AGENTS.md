# Muhtemel Ask Subtitles

Bu depo tek üretim projesidir. Kod `src/mas/`, ayarlar `config/`, testler
`tests/`, RunPod girişleri `runpod/` altındadır.

Önce `README.md`, `docs/ARCHITECTURE_DECISIONS.md` ve
`docs/SEMANTIC_ALIGNMENT.md` dosyalarını okuyun.

## Kurallar

- Önce mevcut kodu okuyun, sonra küçük ve doğrudan değişiklik yapın.
- Kirli çalışma ağacındaki ilgisiz kullanıcı değişikliklerini koruyun.
- Kaynak video SHA-256 kaydından sonra değişmez.
- GPU gereken aşamada CPU fallback kullanmayın.
- Şema, hash, zamanlama, konuşmacı ve çeviri yetkisini test geçirmek için gevşetmeyin.
- Drive teslimini byte sayısı ve SHA-256 readback eşleşmeden tamamlandı saymayın.
- Ağ işlemlerinde sonlu timeout, sonlu retry ve ilerleme watchdog'u kullanın.
- Kimlik bilgilerini Git'e yazmayın.
- Kod düzenlemek veya test çalıştırmak ücretli RunPod izni değildir.
- Gerçek bölüm çalışması dışında GPU başlatmayın.

## Çalışma düzeni

1. `git status --short --branch`
2. İlgili kodu okuyup düzenleyin.
3. Önce hedefli testleri, sonra tam paketi çalıştırın.
4. `git diff --check`
5. Runtime değiştiyse Docker ve RunPod girişlerini kontrol edin.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
.\mas.ps1 doctor --controller
git diff --check
```

Basit çözümü seçin. Tek kullanım için soyutlama eklemeyin. Büyük JSON, altyazı,
log veya medya dosyalarını gereksiz yere açmayın.

## Bağımsız Colab adayı

Kullanıcının 23 Eylül 2026 talebiyle `src/mas/colab_flow.py` ve `colab/`
RunPod gerektirmeyen ayrı bir giriş olarak eklenmiştir. Bu yolda düzeltilmiş
metne yeniden forced alignment uygulanmaz. Önce `colab/README.md` ve
`colab/REVIEW.md` okuyun. `colab/build_notebook.py` ile notebook kopyasını
yeniden üretin; kaynak/notebook eşitliğini test edin. GPU testi yapılmadan
canlı üretim veya akustik doğruluk iddiasında bulunmayın.

## Resmî altyazı yolu

`src/mas/official_subtitles.py` yayıncının mevcut WebVTT zamanlarını kullanır.
ASR çalıştırmaz; sahte kelime zamanları üretmeyin. Önce `automation/README.md`
okuyun. Çeviri mevcut ChatGPT'de yapılır; ücretli API veya başka çeviri modeli
eklemeyin. Yayıncı zamanı korumasını gerçek ses senkron testiyle karıştırmayın.

## Video teslimi (23 Eylül son talep)

`video_flow.py` kaynak indirme, yayıncı altyazısı/ASR seçimi ve mevcut
`engine/burned_mp4.py` GPU gömmesini bağlar. Yayıncı altyazısını bekleme.
`colab_flow.py` geniş ASR + ayrı konuşma aralığı ASR kullanır; iki geçiş aynı
modeldir, bağımsız doğrulayıcı veya dinleme diye sunma. Gerçek T4 testinde gerekli
`preprocessor_config.json` eksikliği düzeltildi. Cuma görevi kayıtlı olsa bile
Colab/Drive başlangıcı ve ses kalitesi doğrulanmadan tam otomatik başarı iddia etme.
