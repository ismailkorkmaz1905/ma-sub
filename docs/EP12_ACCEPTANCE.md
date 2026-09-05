# Bölüm 12 sonrası çalışma kuralları

Durum: `ep12-1` testli güvenilirlik bileşenleri. Tam pipeline release'i değildir.
Kaynak: kullanıcının `EP12_PIPELINE_INCIDENT_REPORT_FOR_ASTRA.md` raporu.

## Hedef

Kaynak doğrulamaya başlanmasından son SRT'nin uzak depoda byte/SHA-256
okumasıyla doğrulanmasına kadar en fazla 240 dakika hedefleniyor. TR ve ID
ChatGPT teslim/bekleme pencereleri de bu ölçüme dahildir. İşlem tekrar
başlatıldığında ilk başlangıç zamanı sıfırlanamaz. Çalıştırma süresi ile
faturalandırılan GPU süresi ayrı ölçülmelidir.

Aşama hedeflerinin tek kaynağı `config/runtime_policy.json` dosyasıdır.
Plan 210 dakika iş/bekleme ve 30 dakika rezerv içerir. Bu rakamlar ölçülmüş
performans veya süre garantisi değildir. Süre hedefini tutturmak için
çeviri kalitesi, immutable alanlar veya akustik doğrulama gevşetilemez.

Bütçenin bitmesi başarılı teslim değildir. Pahalı iş durdurulacaksa checkpoint
ve teşhis çıktısı korunur; QC sonucu ile süre hedefi sonucu ayrı raporlanır.
ChatGPT Pro'nun dönüş süresi programın kontrolünde değildir.

## Raporun ortaya koyduğu zorunlu değişiklikler

- Colab ve notebook runtime yok. Rapordaki Colab fallback önerisi, kullanıcının
  açık "Colab istemiyorum" talimatının önüne geçmez.
- GPU gerektiren aşama CPU'ya sessizce düşmez. GPU/erişim/model cache ön kontrolü,
  uzun ASR işi başlamadan yapılır; GPU kontrol üst sınırı 60 saniyedir.
- Tek bir network retry katmanı vardır. Üç deneme, bağlantı başına 10 saniye,
  veri gelmeyen bağlantıda 30 saniye ve toplam retry üst sınırı 300 saniye.
  Büyük dosya aktarımı için ayrıca byte ilerlemesi ve toplam aşama bütçesi gerekir.
- Kimlik doğrulama, yanlış episode, schema ve hash hataları retry edilmez.
  Geçici bağlantı hatası kullanıcıdan kod düzeltmesi istemeden sınırlı tekrar edilir.
- ASR, audio review ve alignment tamamlanan UID'leri ayrı atomik kayıtlarla yazar.
  Kayıt kimliği input/config/code hash'lerine bağlıdır. Eski correction çıktısını
  yeni input'a sessizce rebind etmek yasaktır.
- TimingOverride sadece zaman yazar; input SHA, UID, eski metin ve eski zaman
  eşleşmeden uygulanamaz. `replacement_text` anahtarı bile reddedilir.
- Uzun cümleyi `Ha?`, `Ne?`, `Ya.`, `Evet.` gibi bir söze indirmek review alarmıdır;
  bu alarm düzeltmenin mutlaka yanlış olduğu anlamına gelmez.
- `clip_start/end` bağlamdır. Coverage yalnız hedef içindeki ölçülmüş konuşmayla
  karşılaştırılır. Boş VAD listesi tek başına non-dialogue kararı değildir.
- Aynı utterance içindeki kelimeler monoton olmalıdır. Doğrulanmış farklı
  konuşmacıların overlap'i ayrı sınıftır; metinler birleştirilmez. Speaker kimliği
  bilinmiyorsa overlap review olarak kalır. Bu sınıflandırıcı mevcut V2 renderer
  veya schema'yı kendiliğinden değiştirmez.
- Production transfer Jupyter/clipboard ile yapılamaz. Final upload geçici isim,
  uzak byte sayısı ve SHA-256 readback, sonra exact isimle yayınlama ister.
- GPU stop kararı kalıcı checkpoint/çıktı doğrulamasından ayrılmaz. Ağ kopması veya
  süreç ölmesi için pod dışından watchdog gerekir. Stop API çağrısı başarısı tek
  başına kapanma kanıtı değildir. Provider state ayrıca okunmalıdır. Storage
  ücreti ile GPU compute ücreti aynı şey değildir.
- NLLB veya başka çeviri modeli otomatik fallback değildir. TR ve ID işleri exact
  ChatGPT paketleriyle yürür. Emergency çıktı strict dosyalarını ve marker'larını
  değiştiremez; strict PASS olmadan formal release sayılamaz.

## Bu değişiklikte gerçekten bulunan kod

`mas.reliability`: kalıcı wall-time bütçesi, sınırlı retry, işlem ağacı timeout'u,
fsync + atomik JSON, input/output/config/code-bound checkpoint, UID günlüğü.

`mas.evidence_guard`: yalnız zaman değiştiren override, stale precondition
kontrolü, semantic shrink alarmı, hedef/bağlam ayrımı, speaker overlap sınıfları.

`mas.progress`: sürekli terminal barı ve tek `status.json`; ölçülmüş iş sayısı,
geçen süre, tahmini kalan süre, son ilerleme ve son checkpoint bilgisi. Toplam
bilinmiyorsa yüzde ve ETA uydurmaz. Yeniden başlatmada eski tamamlanmış işlerden
sahte hız hesaplamaz.

## Doğrulama kapsamı ve kalan işler

Yeni regresyonlar gerçek subprocess kesilmesi/timeout'u, alt süreç kapatma,
checksum bozulması, atomik yazma hatası, checkpoint reuse, yanlış override,
metin bütünlüğü ve ilerleme sayacını çalıştırır.

15 interval, raporun 6.8 bölümünden programatik olarak alınmıştır. Testler bu
koordinatlarda sentetik coverage geometrisini sınar. Raporda yazan sessizlik,
duplicate veya speaker yorumlarını ses dinleyerek doğruladığımız iddia edilmez.
Raporda tam metni olmayan UID için test cümlesi açıkça sentetik veridir.

Drive ZIP'inin dokunulmamış V2 suite'i ayrıca yerelde 413 test ve 145 subtest
ile geçti. Bu, yeni runtime'ın 413 V2 testini port ettiği anlamına gelmez.

Bu branch'teki güvenilirlik bileşenlerinin tamamını eski kısa CLI'a bağlamak,
V2 kaynaklarını normal paket olarak repoya taşımak, gerçek ASR/WhisperX stage
adaptörlerini bitirmek, uzak transfer/RunPod watchdog'u canlı doğrulamak ve
Docker image build'i hâlâ release kapılarıdır. Bu değişiklik tek başına
`./mas run 13` ile gerçek bölüm işleyebilen bir release değildir. `v0.1.0`
etiketi bu commit için oluşturulmaz.
