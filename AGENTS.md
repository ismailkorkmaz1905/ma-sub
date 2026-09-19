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
