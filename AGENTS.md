# Muhtemel Ask Subtitles

Önce `README.md` dosyasını okuyun.

- Üretim kodu yalnız `src/mas/` altındadır.
- Basit çözümü seçin; tek kullanım için yeni modül veya soyutlama eklemeyin.
- Kaynak video kaydedilen SHA-256 değerinden sonra değişemez.
- ChatGPT yalnız `tr` ve `id` metinlerini değiştirebilir.
- Drive teslimini byte sayısı ve SHA-256 readback eşleşmeden tamamlandı saymayın.
- Kimlik bilgilerini Git'e yazmayın.
- Kod düzenlemek ücretli GPU çalıştırma izni değildir.
- Kirli çalışma ağacındaki ilgisiz kullanıcı değişikliklerini koruyun.

Kontrol:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
git diff --check
```
