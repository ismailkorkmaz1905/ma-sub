# Semantik zamanlama

Zaman otoritesi `work/semantic_alignment/word_timeline.jsonl`, dil otoritesi
gerekli olduğunda `handoff/` içindeki ChatGPT ZIP dönüşü, güvenlik otoritesi ise
Python doğrulayıcılarıdır.

Akış:

1. GPU ASR kelimeleri ve zamanlarını üretir.
2. Kesin eşleşen bloklar otomatik kabul edilir.
3. Belirsiz pencereler tek bir hash bağlı ZIP'e yazılır.
4. Dönüşte kimlik, sıra, sahip olunan kelimeler ve metin alanları doğrulanır.
5. Türkçe bloklar aynı kimliklerle Endonezce çeviri paketine girer.
6. QA, MP4 üretimi ve Drive readback tamamlanır.

ChatGPT zaman damgası, kelime kimliği veya blok kapsamı üretemez. Bilinmeyen,
eksik, yinelenen ya da başka pencereye ait kimlikler reddedilir. Konuşmacı ve
sahne sınırları izinsiz geçilemez.

Bir dönüş gerekiyorsa süreç ücretli kaynağı serbest bırakır ve beklenen yolu
ekrana yazar. Aynı `run` komutu dosya geldikten sonra devam eder.
