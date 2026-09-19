# Mimari kararlar

## Tek ürün, tek komut

Üretim girişi `mas run EPISODE` komutudur. Aynı komut yeni çalışmayı başlatır,
ZIP dönüşünde durur ve sonraki çağrıda doğrulanmış checkpoint'ten devam eder.

## Basit bölüm düzeni

- `source/`: değişmez kaynak ve kaynak kimliği
- `handoff/`: dışarı verilen ve geri alınan ZIP dosyaları
- `output/`: yayınlanabilir dosyalar ve teslim makbuzları
- `work/`: yeniden üretilebilir ara dosyalar ve checkpoint'ler
- `.mas/run.log`: son çalışmanın tek yerel logu
- `parts/`: yalnız parçalı teslim kullanılırsa oluşan alt çalışmalar

## Yetki ayrımı

ASR konuşma kanıtını üretir. Semantik aşama kelime zaman çizelgesinden bloklar
kurar. ChatGPT yalnız izin verilen metin alanlarını doldurur. Python kimlik,
sıra, kapsam, zamanlama ve hash bağlarını doğrular.

## Teslim

Yerel dosyanın varlığı teslim değildir. PASS için uzak nesnenin byte sayısı ve
SHA-256 değeri tekrar okunup yerel kayıtla eşleşmelidir. GPU kapanışı dışarıdan
doğrulanmadan yeni ücretli çalışma başlatılmaz.

## Kayıt politikası

Tarih damgalı log arşivi ve e-posta kuyruğu yoktur. Controller kilidi yalnız
aynı anda iki çalışma başlamasını engeller; geçmiş veya olay kaydı tutmaz. İnsan
durumu kısa `status` çıktısından görür, makine ayrıntısı `status --json` ile
alınır. Güvenlik makbuzları yalnız karar kanıtı oldukları yerde tutulur.
