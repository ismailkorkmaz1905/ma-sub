## Alignment follow-up - 19 September 2026

The independently reviewed B1-B4 patch is now applied: noncontiguous conflict
requests bind UID/text/window together, repeated context searches keep scanning,
joint/context word slicing preserves punctuation, and request UIDs are no longer
replaced by merged budget UIDs. Three regression files cover these changes.
The qualification workflow runs the full Python 3.11 suite before committing.
B1/B2 use the explicit `recovery_plan` route; legacy retry permissions are not
automatically upgraded. Existing EP14 evidence needs a freshly validated plan
and exact-commit retry authorization, not an unqualified `run 14` restart.
No episode, paid GPU, live media replay or Drive operation was performed.
This is a code fix, not proof of acoustic resolution of the two remaining UIDs.

# EP14 olay sonrası kod incelemesi ve sınırlı onarım

Tarih: 19 Eylül 2026, Singapur.

## Karar

**Üretime hazır / bütün sorunlar kapandı kararı verilmedi.** Bu çalışma alignment
recovery kapsamını açıkça planlayan bir yol, model çağrısı bütçesi, yeniden kullanım
regresyonları ve metadata kontrolleri getirir. Tek-component hybrid teslim,
deadline-aware orchestration ve birleşik retry state machine tamamlanmış değildir.

İncelenen olay sonu üretim commit'i:
`8a54ac3b095fa6635d823a7558ce20dccfb8dc24`.
Kaynak aktarımı için yalnızca audit-snapshot workflow'una isteğe bağlı revision
alanı eklendi: `b6eae83587514c7224cbb4beee3a772193d2c464`.
Alınan kaynak ağacı `f5d6fb7b364307d5a3a45d279e10732696323313` olarak doğrulandı.

## Kanıt sınırı

Kullanıcının `Pasted text.txt` dosyası 17 saat 17 dakika 1 saniyelik olayın,
44 çalıştırmanın, 36 commit'in, 106 kredi satırının ve son teslimin raporudur.
Bu inceleme bu sayıları bağımsız ham log sayımı olarak sunmaz.

GitHub kaynakları ve testleri doğrudan incelendi. Gerçek `EPISODES/` çalışma
klasörü, başarısız CTC çağrılarının ham çıktıları ve son readback receipt'i bu
kaynak arşivinde yoktur. Bağlı Drive'da arama ve teslim klasörü listesi yapıldı;
son emergency MP4 bulundu, sağlayıcı metadata boyutu 5.184.093.433 byte olarak
raporla eşleşti. Dosya ID'si `1qR1zur0DZkmLjUm0eIY-hsKSZsSgKnvU`.
Metadata okuması tam SHA-256 readback değildir. Video yeniden indirilmedi,
izlenmedi ve ses dinlenmedi. Bu çalışmada GPU/RunPod başlatılmadı, Drive'a yazılmadı,
e-posta gönderilmedi ve bölüm yeniden işlenmedi.

106 satırlı yeni fixture sentetiktir. Sayısı rapordaki olayı temsil eder; gerçek
EP14 UID'lerini, metinlerini veya zaman dağılımını yeniden ürettiği iddia edilmez.

## Kodda doğrulanan temel bulgular

### P0 - Yayın kapsamı ile kalite yolu birbirine bağlı

`src/mas/pipeline.py:run` içindeki `first-hour-v1` dalı progressive işleyiciye
geçer. `whole-episode-v1` aynı delivery-first işin tek dosyalı yayını değildir;
strict TR correction, audio review ve forced alignment zincirine gider.
`src/mas/progressive.py` içindeki delivery-first seçimi ayrıca scoped retry
varlığına bağlıdır. EP14 delivery-first sözleşmesi tek başına bütün giriş
noktalarının aynı politikayı kullandığını garanti etmez.

Bu, olay raporuna ek bir kod bulgusudur. Gerçek komut/env logları olmadan
operatörün hangi geçişi ne zaman yaptığı bu incelemede kanıtlanamaz.

**Açık düzeltme:** yayın kapsamı ve kalite politikasını ayrı, kalıcı ve
kimlik doğrulamalı run contract alanları yap. Eski episode state'ini sessizce
delivery-first'e çevirmek veya strict receipt'i yeniden etiketlemek yasak kalmalı.
Bu onarım yönlendirmeyi değiştirmedi.

### P0 - Dinamik arama ile dar retry kapsamı uyuşmuyor

`forced_align.py` içinde recovery komşu context'e geçebiliyor. Eski authorization
UID ve component limitlerine dayanıyor. Joint çağrılar context UID'lerini
`_mas_component_uids` içinde taşıyor; bunları yalnızca ilk hedeflerin altkümesi
olarak kabul etmek, doğru metin ve ses aralığına sahip bir isteği de reddedebiliyor.
Bir sonraki component'e izin eklemek bu problemi genel olarak çözmüyor.

**Uygulanan yol:** yeni `alignment_recovery.py` aynı doğrulanmış source üzerinden
bağlı bölgelerin tamamını baştan çıkarır. Coarse aralıklar arasındaki boşluk
1.000 ms'yi aşıyor ve kaynak alignment pencereleri de ayrılıyorsa bölge kesilir.
Hedeflere bağlı bölgenin tamamı planda görünür. Uzun bir sahne büyük closure
üretebilir; bu gizli ve otomatik yetki genişlemesi değildir.

Plan source hash'i, ilk hedefler, kesin component UID dizileri ve operatörün
seçtiği yeni CTC çağrısı üst sınırını taşır. Plan yeniden hesaplanarak doğrulanır.
Bir isteğin UID'leri tek yetkili component içinde, ses aralığı onun zarfı içinde,
metni de aynı component'in ardışık kaynak metniyle birebir aynı olmalıdır.
NaN, sonsuz, boolean zaman, yinelenen UID ve kaynak dışı metin reddedilir.
Mevcut kelime, score, drift, overlap ve strict finalization kuralları değişmez.

Eski izinler genişletilmez. Yeni `recovery_plan` eski discovery/continuation
limitleriyle aynı izinde kullanılamaz. Yeni plan existing code-fix permit
mekanizmasından ayrıca yetkilendirilmelidir; CLI yalnızca teklif yazar.

### P1 - Yeniden başlatma ve başarılı model çağrıları

Başarılı ham CTC çağrıları için `UnitJournal` olaydan önce de vardı. Dolayısıyla
"hiç checkpoint yok" teşhisi yanlıştır. Yeni yol bu cache'i korur.

Yeni çağrı yapılmadan önce request hash'i HMAC ile doğrulanan kalıcı bir bütçeye
eklenir. Çökme sonrası tamamlanmamış çağrı harcanmış deneme sayılır; tekrar
kalan bütçeden düşer. Cache hit yeni çağrı bütçesi tüketmez. Aynı scope ile
restart, exact arşivlenmiş conflict predecessor ve ham cache hash'leri
üzerinden devam edebilir; `latest-conflict-failure.json` dosyasını elle geri
koymak gerekmez.

Bu, tek controller/worker altında atomik yerel kayıt modelidir. Dış rollback
koruması, dağıtık eşzamanlı yazıcı desteği veya bütün retry state machine'in
birleştirilmesi olarak sunulmaz. Eski dinamik discovery receipt'leri signed
append-only graph'a dönüştürülmedi. Yeni yol runtime graph genişletmek yerine
önceden yetkili sabit closure kullanır.

### P1 - Resolver kimliği sabit bir değerdi

`ALIGNMENT_RESOLVER_PRODUCER_SHA256` sabitti. Component/normalized cache'in
geçerliliği gerçek resolver değişikliklerini otomatik olarak yansıtmıyordu.
Artık bu türetilmiş cache'ler gerçek `forced_align.py` dosya hash'ine bağlıdır.
Ham model çağrısının producer hesabı değiştirilmedi. Böylece yeni validator
çalışırken daha önce hesaplanmış uygun ham CTC sonucu yeniden kullanılabilir.
Yeni helper dosyaları ilgili stage/finalization producer listelerine eklendi.
Eski final marker'ları yeni producer'a otomatik taşınmaz.

### P1 - Eski partial hata state'i whole-episode kimliğine sızabiliyor

`runpod_controller._partial_failure_identity` şimdi whole-episode modunda
partial pointer'ı okumadan boş sonuç döndürür. Bozuk veya eski part pointer'ı
whole-run hata kimliğine ekleyemez.

Bu belirli sınır düzeltildi. Bütün checkpoint, export, return ve receipt
namespace'lerinin yeniden tasarlandığı iddia edilmez. Kod değişince eski
export producer kontrolünün reddetmesi kaldırılmadı; güvenli migration
sözleşmesi hâlâ ayrı bir açık iştir.

### P1 - Kredi metni kontrolleri ve tekrar dağılımı

Olay sonu baseline'da bazı kaynak ve ID-return kredi kontrolleri zaten vardı.
Bu nedenle tarihsel "hiç kontrol yoktu" iddiasını mevcut HEAD üzerinden
tekrarlamak doğru değildir. Strict translation pack/return sınırları ve bazı
Unicode/case/K.M. varyantları güçlendirildi.

Yeni ortak detector yalnızca bilinen tam kredi satırlarını yakalar. Örneğin
`Altyazı M.K.`, `Takarir M.K.`, `TAKARIR K.M.` ve full-width/format-karakterli
varyantlar. `Altyazı M.K. yazıyor.` gibi daha uzun gerçek metin otomatik silinmez.

Delivery-first kaynak yolunda bu metadata karantinaya alınır, kaynak değiştirilmez.
Strict translation pack oluşturma/doğrulama ve ID-return doğrulama kirli kaydı
reddeder. Tekrar QA aynı normalize metnin sayısını, UID'lerini, ilk/son zamanını
ve dakika kovalarını raporlar. `Evet.` gibi gerçek tekrarları yalnızca sayısı
nedeniyle silmez. Correction-pack öncesi bütün kaynak yollarını tek bir yeni
karantina modeline taşımak bu patch'in kapsamı dışında kaldı.

## 17 talebin durumu

| No | Talep | Bu onarımın gerçek kapsamı |
| --- | --- | --- |
| 1 | Dinamik keşif / statik yetki uyuşmazlığı | Yeni, açıkça yetkilendirilen closure yolu uygulandı. Legacy permit otomatik genişletilmez. |
| 2 | Deterministik component closure | Kaynak geometrisinden bağlı bölge planı ve exact doğrulama eklendi. Gerçek EP14 closure ölçülmedi. |
| 3 | Signed append-only runtime graph | Yeni yol runtime discovery kullanmaz. Legacy graph dönüşümü yapılmadı. Kalıcı çağrı bütçesi HMAC korumalıdır. |
| 4 | Başarılı CTC'yi tekrar çalıştırmama | Gerçek alignment wrapper + fake model sınırında cold/warm/restart ve resolver değişimi test edildi. |
| 5 | Tek-component coarse fallback | AÇIK. Delivery-first per-cue fallback mevcut; strict-success koruyan hybrid receipt/finalizer yok. |
| 6 | Partial/whole namespace ayrımı | Hata kimliğine partial sızması düzeltildi. Tam namespace migration AÇIK. |
| 7 | Runtime/Drive/ownership preflight | Mevcut controller doctor incelendi. Tam canlı image/model/ownership qualification eklenmedi. |
| 8 | GPU öncesi local dry-run | Sentetik/offline testler var; gerçek hesap kapasitesi veya GPU ABI'si ücretsiz local testle kanıtlanmadı. |
| 9 | Çeviri öncesi metadata kontrolü | Translation pack create/read, ID return ve delivery-first kaynak sınırı güçlendirildi. TR correction öncesi tüm yollar tamamlanmadı. |
| 10 | Frequency/time-distribution QA | Uygulandı; rapordaki sayıyı temsil eden 106 satırlı sentetik regression eklendi. |
| 11 | Stage-local recovery planı | Offline teklif + exact scope + kalıcı çağrı bütçesi eklendi. Genel otomatik recovery tamamlanmadı. |
| 12 | Birleşik retry state machine | AÇIK. Eski code-fix permit ve predecessor kontrolleri korunuyor. |
| 13 | Deadline-aware seçenek ve ETA | AÇIK. Encode/upload/readback ölçümleriyle kritik yol kararı henüz yok. |
| 14 | Kimliği sabit context/overlap çeviri parçaları | `translation_workspace.py` zaten owned UID ve read-only context/hash sözleşmesi içeriyor. Yeni paralel paket sistemi eklenmedi. Emergency/whole entegrasyonu ayrıca gerekir. |
| 15 | İsim/dini ifade/sıra/anlam QA | Metadata eklendi; mevcut immutable/order/policy kontrolleri korundu. Yeni komşu anlam değerlendirmesi yapılmadı. |
| 16 | EP14 integration replay | Hedefli sentetik wrapper/controller/pack testleri eklendi. Gerçek 44-run olay replay'i yapılmadı. |
| 17 | Hash/readback/source güvenliğini koruma | Kontroller kaldırılmadı. Yeni yardımcı dosyalar producer kimliğine dahil edildi; strict/emergency ayrımı korundu. |

## Operatör işlemi

Yeni komut yalnızca mevcut yerel kanıttan bir teklif üretir:

```powershell
.\mas.ps1 plan-alignment-recovery 14 --max-new-ctc-calls 64
```

`64` burada örnek bir **yeni model çağrısı sayısıdır**, önerilmiş GPU bütçesi veya
gerçek EP14 ihtiyacı değildir. Part için `--part-id part-001` gerekir.
Çıktı `review/alignment-recovery-plan.json` altında `PROPOSAL_NOT_AUTHORIZED`
olur. Eksik/eski/uyuşmayan kanıt hata verir; episode çalıştırılmaz, mail gönderilmez.

Planın closure kapsamı ve çağrı sınırı operatör tarafından incelenmeli; mevcut
`authorize_code_fix_retry` yolu exact current commit, failure/predecessor ve
başarılı bound fixture kanıtını doğrulamadan paid retry yapılmamalıdır. Bu patch
eski EP14 izinlerini, deadline'ı, anahtarı veya raw-ASR marker'ını yeniden yazmaz.
Şu anda çalışan bir bölüme doğrudan `git pull` ve tekrar `run` önerilmez.

## Açık işlerin kabul ölçütleri

**Hybrid tek-component teslim:** Başarılı strict UID/kelime sonuçları aynen
korunmalı. Yalnızca exact source/text/alignment hash'lerine bağlı açık onaylı
component coarse timing almalı. Komşu cue çakışması ve okuma hızı tekrar
kontrol edilmeli. Receipt UID, zaman aralığı, neden ve onayı taşımalı;
`strict_eligible: false` olmalı ve strict marker üretmemeli. Yetkisiz component,
değişmiş source veya yeni overlap reddedilmelidir.

**Deadline orchestration:** Kalan kritik yol, çeviri bekleme + validation + encode
+ upload + readback + pay olarak hesaplanmalı. Raporun yaklaşık 63 dakikalık
fiziksel son yolu gelecekteki garanti değil, gözlemdir. Yeni ölçüm yoksa ETA
belirsizliği görünmeli. Bütçe uzatma veya kalite politikasını değiştirme sessiz
olmamalı; mevcut deadline hiçbir restart ile sıfırlanmamalıdır.

**Recovery state machine:** Kanıt değişimi, code-fix, superseded review ve transfer
retry ayrı izin dosyalarının rastgele kombinasyonu yerine tek geçiş sözleşmesine
bağlanmalı. Her geçişin invalidation cone'u açık olmalı: yalnızca gerçekten
etkilenen stage yeniden hesaplanmalı. Yetki ve producer kontrolleri kapatılmamalı.

**Qualification:** Gerçek retained EP14 kanıtıyla yeni closure'ın büyüklüğü,
kaç uncached çağrı gerektirdiği ve iki unresolved UID'nin çözümü ölçülmeli.
Ayrı açık yetkili GPU testi olmadan acoustic PASS, dört saatlik SLA veya üretim
hazırlığı ilan edilmemelidir. Tam bölüm perceptual review ayrıca gerekir.

## Test kanıtı

Değişiklik öncesi tam yerel suite: Python 3.13.5 üzerinde 1.455 test ve
175 subtest geçti, 109,14 saniye. Bu target Python 3.11 qualification değildir.
Ara bir full-suite çalışırken kaynak dosyası değiştiği için bir frozen-producer
teardown kontrolü hata verdi; o çalıştırma final doğrulama sayılmadı ve kontrol
gevşetilmedi. Son patch sabitlenerek tüm suite yeniden çalıştırılır; kesin sonuç,
tested tree ve Python 3.11 CI çıktısı teslim edilen doğrulama dosyaları/CI logunda
kaydedilir. Model/provider sınırları testlerde sahtedir; gerçek CPU FFmpeg
sentetik medya testleri bunun dışında çalışır.
