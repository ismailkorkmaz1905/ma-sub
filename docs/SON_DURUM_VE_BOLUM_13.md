# Son yaşananlar ve Bölüm 13

**Bölüm 12 MP4 ve Drive teslimi tamamlandı.** [MP4'ü Drive'dan aç](https://drive.google.com/file/d/1FAkmN0Z9HWcRrbfOT8C-ftmSXXiDdA2I/view?usp=drivesdk).

1080p, Türkçe ses, görüntüye gömülü Endonezce altyazı. Dosya yaklaşık 8,98 GB. Tamamı indirilip kontrol edildi; Drive'daki son adıyla da tekrar okunarak aynı olduğu doğrulandı. Bilgisayardaki dosya: `C:\Users\Ismail\CodeBase\ma-sub-archive-20260906\deliverables\Muhtemel Ask 12.Bolum.id.BURNED.REVIEW.mp4`.

## Son üç gün: 4, 5 ve 6 Eylül 2026

Tarihler Singapur saatine göredir. Aşağıdaki sıra Git kayıtlarına ve gerçek teslim kayıtlarına dayanır; bütün sürecin dört saatte bittiği iddia edilmiyor.

**4 Eylül - Projeyi kurmaya uğraştık.** İlk depo, komut satırından çalıştırma ve kaynak kodu taşıma işleri yapıldı. Taşıma paketlerinde bozulma ve kurulumu kaldığı yerden sürdürme sorunları çıktı; peş peşe onarım kayıtları oluştu. Bu günün işi ağırlıklı olarak sistemi ayağa kaldırmaktı. Dayanak: `a12ff1b`, `f703a50`, `02585d3`, `7bf8a4f`.

**5 Eylül - Bölüm işini yöneten programı toparladık.** Üretim kodu tek projede toplandı, `main` ana dal oldu. Uzak makineyi açma, ilerlemeyi gösterme, kesilen işi sürdürme ve kapanışı kontrol etme eklendi. Bölüm 11 denemelerinde indirme, bağlantı ve altyazı zamanlaması sorunlarıyla uğraşıldı. Şüpheli sözcükleri sırf çıktı oluşsun diye kabul etmek yerine ses incelemesine ayıran kontroller eklendi. Bu, tamamlanmış kalite onayı değildi. Dayanak: `95928e3`, `538f9ce`, `09320be`, `3cbad05`, `1465c27`, `26d66f9`.

**6 Eylül - Gerçek çıktıya ulaştık.** Bölüm 11'in kısa pilotu üretildi; kalite durumu inceleme gerektiriyordu. Senin sonraki talimatınla Bölüm 12'ye geçildi. Türkçe ve Endonezce inceleme altyazıları, ardından 1080p gömülü altyazılı MP4 tamamlandı. Son dosya Drive'dan iki kez tamamen okunarak doğrulandı. Geçici GPU silindi. Kod ve belgeler kaydedildi; proje içindeki büyük çalışma dosyaları dışarı ayrıldı. Bölüm 13 başlatılmadı.

### Bugün bizi en çok ne yordu?

1. **Altyazıyı ürettik.** Bölüm 12 için Türkçe ve Endonezce dosyalar var. Metnin kaybolmaması ve dosyaların bozulmaması kontrol edildi. Ama bazı hızlı cümleler, zamanlamalar ve kimin konuştuğu hâlâ dinleyerek kontrol edilmeli. "Kusursuz bitti" diyemiyoruz.
2. **Bilgisayarı fazla yükledik ve durdurduk.** Yerel dönüştürme ile testleri aynı anda çalıştırmak hataydı. Sonra bağlantı ayarını düzelttik; uzak makinede aynı dosya kontrolleri takılmadan tamamlandı. Yeni yerel dönüştürme yapılmadı.
3. **MP4'ü uzak ekran kartında tamamladık.** Kalıcı diskin kotası dolunca ilk üretim yarım kaldı. Kurulum artıklarını temizleyip aynı makinenin boş geçici diskini kullandık. Başarılı tam dönüşüm yaklaşık 8 dakika sürdü; bütün teslim süreci bu kadar kısa değildi. Google'ın ortak bağlantı kotası yükleme sonrası kontrolü de aksattı. Dosyayı tek bağlantıyla indirerek doğruladık, adını düzelttik ve son hâlini tekrar kontrol ettik. MP4 artık hem Drive'da hem bilgisayarda hazır.

## Ana programda ne değişti?

2026-09-07 güncellemesi: Bölüm 13 henüz yayımlanmadı. Yaklaşık 3 GB boyut
hedefi artık ana koda bağlı, kesin tavan değil. Geçici Pod seçimi, tek iş
başlatma, bağlantı kopunca durumdan izleme ve örnek sahne kapıları eklendi.
Güncel davranış, testler ve gerçek koşu önkoşulları
[hazırlık raporunda](EP13_CODE_PREPARATION_2026-09-07.md). Aşağıdaki eski test
sayıları ve henüz uygulanmamış ayar notları tarihsel kayıttır.

- Son teslim artık Endonezce yazısı görüntünün içinde olan MP4 olacak. Türkçe ses ve kaynak çözünürlüğü korunacak.
- Uzak bağlantı terminalden giriş beklemeyecek. AV1 görüntünün açılması da ekran kartında yapılacak; yalnız son kodlama değil.
- Yazı sade beyaz, ince siyah kenarlı ve ekranın alt ortasında olacak. Bu seçilmiş bir görünüm; bütün sinemaların tek zorunlu standardı olduğu iddia edilmiyor.
- Eski kalite kontrolleri kaldırılmadı. Kaynak ve çeviri dosyalarının değişmediği kontrol edilecek.
- Drive'a yükleme tek başına yeterli sayılmayacak. Dosya geri okunup sağlam olduğu doğrulanacak.
- Son tam yerel test: **742 başarılı, 5 atlandı**. Bu, Bölüm 12'yi dinleyerek onayladığımız anlamına gelmiyor.

## Dosyalar nerede?

Ana proje `C:\Users\Ismail\CodeBase\ma-sub` içinde. Eski bölüm çalışmaları ve gerekli kayıtlar `C:\Users\Ismail\CodeBase\ma-sub-archive-20260906` içine ayrıldı. Eski belgelerdeki `var/` ve `EPISODES/` yolları artık bu arşivin altındadır.

Son Bölüm 12 altyazıları ve altyazı seçilebilen MKV, arşivde `EPISODES/Muhtemel Ask 12.Bolum/work/review-delivery-20260906-final-utf8/` içindedir. Bunlar inceleme sürümüdür. Tam MP4'ün gerçek üretim kayıtları aynı arşivde `mp4-final-20260906T125824Z/` içindedir. Başlangıç, orta ve sonlara yakın sahnelerde yazının görüntüye işlendiği görsel olarak kontrol edildi; bu, ses dinleyerek kabul değildir.

405 geçici dosya, toplam 5,66 GB, geri dönüşüm kutusuna kaldırıldı. Bunlar ses kopyaları, yarım videolar ve model önbellekleri. Kutuyu boşaltmadan disk alanı tamamen geri kazanılmaz. Kimlik bilgileri ve çalışan Python ortamı korundu.

**Drive teslim klasöründe artık yalnız bir MP4 var.** [Muhtemel_Ask_Subtitles klasörünü aç](https://drive.google.com/drive/folders/1Gdn4WLjICGJNsYIMCSpSNyXgA_8mdL5p). MP4 doğrudan bu klasöre taşındı; bağlantısı, boyutu ve içeriği değişmedi. Canlı klasör listesinde başka dosya veya alt klasör yok.

Eski MKV/SRT'yi çöp kutusuna taşıma isteği Google'ın ortak bağlantı kotasında HTTP 403 aldı; tarayıcı erişimi de kurulamadı. Bu yüzden eski klasör silinmedi, teslim klasörünün dışına, Drive ana dizinine **Muhtemel Ask - Eski teslim arsivi - 20260906** adıyla taşındı. Bu geri alınabilir düzenleme teslim klasörünü temizledi; Drive'daki eski dosyaların kapladığı alanı boşaltmadı. [Eski arşiv](https://drive.google.com/drive/folders/1baEW_vbsemXqM-K1or6SOEAyP63DCGpI). Tam silme yerine yapılan bu işlem özellikle kaydedildi.

Uzak diskte pip/uv kurulum önbellekleri ve bu denemenin yarım MP4'ü kaldırıldı. Kullanılan modeller, çalışma ortamı ve kaynak video korundu. Kalıcı disk silinmedi.

Kalıcı toplu silme otomatik güvenlik denetiminde engellendiği için geri alınabilir temizlik yapıldı. Windows ve Drive çöp kutuları boşaltılmadı. Silinen dosya listesi ve Drive karşılaştırma kaydı arşivde saklandı.

## Süre ve bölüm başı maliyet

İlk Bölüm 11 denemeleri için kullanıcı yaklaşık **21 saat** ve **20 USD** bildirdi. Sağlayıcı faturası olmadığı için bu iki sayı kullanıcı raporudur. Çok sayıda yeniden başlatma, bozuk kurulum, kapasite, SSH ve tekrar çalışma bu toplamı büyüttü.

Mevcut sistemde yeni bölüm hedefi **toplam 2-4 saat**. Bu süre garanti veya kanıtlanmış hizmet seviyesi değildir; Bölüm 12'deki gerçek aşama ölçümleri ve giderilen hatalar üzerinden planlama aralığıdır. Çeviri dönüşü, Drive kotası veya yeni bir hata bekletirse uzayabilir.

| Kalem | Ölçüm veya hesap | Bölüm başı tahmin |
|---|---:|---:|
| Kaynak indirme ve ses hazırlama | Bölüm 12'de 125,786 saniye | GPU açık tutulursa yaklaşık 0,02 USD |
| Tam Whisper ASR | Transfer ve worker 1.186,547 saniye | L4 için yaklaşık 0,16 USD |
| CTC hizalama | 425,802 saniye | L4 için yaklaşık 0,06 USD |
| Hedefli ASR/CTC onarımları | 375,868 saniye | L4 için yaklaşık 0,05 USD |
| 1080p gömülü MP4 | 481,949 saniye | L4 için yaklaşık 0,07 USD |
| Yerel konuşmacı analizi | 8.743,941 saniye, yerel CPU | 0 USD RunPod GPU |
| 50 GB kalıcı disk | 0,07 USD/GB/ay | 3,50 USD/ay; 2-4 saatlik koşuya yaklaşık 0,01-0,02 USD düşer, fakat bölüm yokken de yazmaya devam eder |
| Drive, YouTube ve Gmail | Bu koşuda kullanım/kota gözlendi | Kanıtlanmış bölüm başı ek ücret 0 USD; kota sınırları süreci geciktirebilir |
| Codex Astra/Sol | ChatGPT/Codex kullanım kotası | RunPod faturasına dahil değil; bölüm başı kesin haftalık kota yüzdesi bilinmiyor |

Hesapta son gerçek pilotta görülen **L4 0,49 USD/saat** fiyatı kullanıldı. ASR, CTC, hedefli onarım ve MP4 üretiminin ölçülen toplamı yaklaşık 41 dakika ve teorik çıplak karşılığı yaklaşık **0,34 USD**. Kaynak hazırlığı GPU açıkken yapılırsa yaklaşık 0,02 USD daha eklenir. Pod açılışı, dosya aktarımı, bekleme, kısa testler ve hata payı bunun üstüne gelir. Bölüm 12'nin tam ASR başlangıcı ile son dış gözlem arasında hesap bakiyesi **3,6165304786 USD'den 2,8764311039 USD'ye**, yani yaklaşık **0,74 USD** düştü; bu bir fatura dökümü değil, aynı hesaptaki iki bakiye gözlemidir.

Yeni bölüm için pratik planlama bütçesi **1-2 USD**. L4 dört saat boyunca hiç kapatılmazsa yalnız GPU yaklaşık **1,96 USD** tutar; bu yüzden çeviri veya insan bekleme sırasında GPU açık bırakılmamalı. Son bakiye eski olduğu için her koşudan hemen önce fiyat ve bakiye yeniden sorgulanmalı; 1 USD ve kapanış payı korunmalı.

## Cuma başlamadan önce

**Son tercih, 6 Eylül:** Bölüm 12 mevcut hâliyle kalacak; yeniden sıkıştırma yapılmayacak. Drive'daki adı YouTube başlığıyla eşleşen `Muhtemel Aşk 12. Bölüm.mp4`. Sonraki bölümlerde öncelik yüksek görüntü kalitesi ve 1080p; bölüm başına yaklaşık 3 GB hedefleniyor. Bu kesin üst sınır değil: boyuta yetişmek için belirgin görüntü bozulması kabul edilmeyecek. Uygun sıkıştırma, bölüm süresi ve kısa sahne denemelerine göre seçilecek. Bu tercih şimdilik not ve operatör talimatıdır; ana kodun boyut ayarı henüz değiştirilmedi.

- **YouTube çerezini yenile.** Cuma hâlâ çalışacağı garanti değil.
- **Drive bağlantısını kontrol et.** Ortak Google bağlantısında kota hatası yaşandı. Kendi Google bağlantı kimliğine geçmek gerekiyor; kullanıcı hesabıyla yetkilendirme gerekebilir.
- **Gmail'i bir kez dene.** Bugün günlük gönderim sınırına takıldı. Bildirimlerin gittiğini varsayma.
- **Bakiye ve bağlantıyı kontrol et.** Son görülen bakiye yaklaşık 2,88 USD; tam değer 2,8764311039 USD (6 Eylül 14:01 UTC). Geçici GPU'nun silindiği dışarıdan doğrulandı; eski makine kapalı. Saklanan disk ücret yazmaya devam ediyor: son gözlem 0,005 USD/saat. En az 1 USD ve kapanış payı korunacak.
- **Önce küçük uzak dönüşüm denemesi.** Bağlantı ve örnek MP4 gerçekten tamamlanmadan uzun iş başlatılmayacak. Yerel bilgisayarda ağır dönüşüm yapılmayacak.

Masaüstündeki **Muhtemel Ask - Episode 13 Astra** kısayolu güncel promptu okuyacak. Başka bölüm için `tools/Start-Episode.ps1 -Episode X` içindeki sayıyı değiştirmen yeterli.

**Bölüm 13 başlamadı.** Başlatıcı ve kod hazırlandı; cuma erişim, bakiye ve disk alanı yeniden kontrol edilmeli. Google'ın ortak bağlantı kotası ve dinleyerek altyazı kontrolü hâlâ dikkate alınmalı. MP4 dosyasının tamamlanması, özgün 11 aşamanın hepsinin kalite kabulü aldığı anlamına gelmez.

## Kaydedilen değişiklikler

- `9332295`: bölüm çalıştırma ve altyazı/hizalama düzeltmeleri.
- `0585230`: önceki raporlar ve Episode X başlatıcısı.
- `00b9f66`: kalite kapılarından sonra gömülü altyazılı MP4 üretimi ve MP4 teslimi.
- `697f5d2`: otomatik uzak bağlantının giriş beklemesini önleyen ayar.
- `047d1c8`: AV1 görüntüyü ekran kartında açarak dönüşümü hızlandırma.
- `156b4ca`: gerçek MP4 teslim kanıtları, temizlik ve Bölüm 13 hazırlığı.
- Bu üç günlük rapor ve yalnız MP4 kalan Drive düzeni sonraki belge commit'inde kaydedilir. Kesin son kayıt `git log -1 --oneline` ile görülebilir; masaüstündeki commit listesinde de bulunur.

Son tam test 742 başarılı, 5 atlandı; 57,59 saniye sürdü. Yeni Docker görüntüsü oluşturulmadı. Tam bölüm MP4 üretimi gerçek L4 ekran kartında tamamlandı. Drive'dan iki tam okuma yapıldı; 8.978.040.872 byte ve SHA-256 eşleşti. Kanıt: arşivde `mp4-final-20260906T125824Z/recovery-complete.json`. Bu ayrı inceleme koşusu, tüm strict akışın gerçek bölüm kabulü değildir.
