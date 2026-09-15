# Pipeline uygulama kaydi - 14 Eylul 2026

## Guncel paket: Episode 14, ilk saat oncelikli uretim

Bu bolum onceki paket notlarinin guncel durumudur. Onceki zaman denetimi yeniden yapilmadi; Episode 13 dosyalari degistirilmedi. Bu turdaki kanitlar kod, yerel testler ve mevcut kayitlardan gelir. Gercek Episode 14, ucretli RunPod veya canli Drive islemi yapilmadi.

| Konu | Uygulanan davranis | Kanit ve sinir |
|---|---|---|
| Yerel yakma | Varsayilan controller yolu Intel QSV. GPU sadece ASR/ses incelemesi/hizalamada; yakma ve yukleme disaridan dogrulanmis Pod kapanisindan sonra. | Bu bilgisayarda Intel N150, 4 cekirdek/4 thread, 15,8 GiB RAM, Intel Graphics; tek-kare QSV testi PASS. Eski yerel EP13 burn receipt: 8855,16 saniyelik 1920x1080 kaynak 1940,25 saniyede (32 dakika 20,25 saniye) h264_qsv q18 ile yakilmis. Bu, yeni EP14 hizi veya eski emergency altyazi kalitesi icin PASS degildir. |
| Hizalama israfi | Ayni degismeyen hata tekrar GPU isi acamaz. Degisen kod icin yerelde gecmis fixture, exact component yetkisi, degismeyen onceki ASR/TR/ses-inceleme kanitlari gerekir. Cozulmus adaylar, deterministik retler ve component secimleri yeniden kullanilir. | Onceki EP13 auditinde 23 basarisiz hizalama 22765,7 saniye; 5 bitmemis denemenin son heartbeat alt toplami 4867,5 saniye. Toplam en az 27633,2 saniye (7 saat 40 dakika 33,2 saniye); faturalandirilan GPU suresi degildir. |
| Hizalama secicisi | Iki kopya secici tek exact secim yolunda; sabit komsu ve aday cifti uyumu bir kez hesaplanir. Yalniz mevcut en iyi sonucu matematiksel olarak gecemeyecek dallar elenir. Bilinen farkli konusmaci overlap'i ve komsu olmayan kelime cakismalari korunur. | 200 tohumlu kucuk component x 2 secim hedefi eski exhaustive sonucuyla ayni. Sentetik is sayaci fixture'inda 7776 tam-deneme rebuild yerine 0; en fazla 120 cached pair kontrolu. Bunlar saniye veya GPU hizlanma olcumu degildir. |
| Sinirli kurtarma | Component icin 120 saniye / 64 yeni inference cagrisi; kurtarma toplaminda 900 saniye. Parent VAD ve turetilmis ses islemi ayri 600 saniye tavanli, probe 60 saniye tavanli; hepsi kalan bolum saatinin icinde. | GPU cagrisi icindeki kilitlenmeye karsi dis worker watchdog'u da kalir. Kurali gecirmek icin anlam/senkron/VAD QA kapatilmaz. |
| Ilk saat | Controller varsayilani `first-hour-v1`. Tam kaynak/ses ve bagimsiz VAD planindan sonra ASR, TR, ses inceleme, hizalama ve ID yalniz ilk yayinlanmamis parca icin calisir. Ilk parca yerelde yakilip Drive byte/SHA readback dogrulanmadan sonraki parcanin ASR'i baslamaz. | Gercek child ses PCM/sample soyagi, tum parent VAD bolgelerinin kapsanmasi, dondurulmus TR/ID paketi ve strict partial raporu zorunlu. 1 saat hedef kesimidir; diyalog kesilmez, gercek guvenli sahne araligi raporlanir. Kesintisiz konusmada tam 60:00 garanti edilmez. |
| Kesintiden devam | TR/ID bekleme25, strict partial encode26, sonraki parca27. GPU kapanisi ile yerel is arasindaki kesinti, ayni parcayi yeniden GPU'ya gondermez. Eksik controller ACK, dogrulanmis teslimden tekrar uretilir. | Her parcanin degismeyen exportu, kapanis kaniti, encode ve Drive receipt'i ayridir. Sonraki hata ilk parcayi silmez/yeniden yayinlamaz. |
| Parca sonu | Kesim guvenli VAD/caption boslugunun sonuna yaklastirildi; sonraki konusmaya mevcut 120 ms akustik tolerans birakilir. Son sample siniri, ID paketi dondurulmadan once altyazi gruplama/secimine girer. | Yalniz opsiyonel gosterim payi kisalabilir. Gercek son kelime, minimum sure ve ayni CPS kosulu korunur; sigmiyorsa tam UID ile erken hata verilir. Son ornekteki kesirli milisaniye icin VAD tavan yuvarlamasi duzeltildi, kelime zamanlari gevsetilmedi. |
| Tamamlanma | Her planned sample araligi ayri MP4 ile teslim edilir; toplu durum `COMPLETE_PARTS`. | Tek bir birlestirilmis tam bolum videosu veya eski full-episode PASS diye sunulmaz. Tum parcalarin yerel MP4'i ve uzaktan byte/SHA kaniti olmadan tamamlanma yoktur. |
| Ceviri/senkron | Kaynak kelimelerin cue/sahne sahipligi, gercek ilk kelimeden once baslamama, son kelimeden once bitmeme, eksik konusma/VAD, dogal ID review, Allah/dini ifade kurallari korunur. | Otomatik kuralli testler insanin duydugu serbest anlam/dogallik ve gercek video senkronunun tam ispati degildir. Onceki 13. bolum emergency ciktilari aynen kalir. |
| Drive | Native resumable session, dogrulanmis offset'ten eksik-byte devam, exact kaynak/hedef/OAuth kimligi, bounded retry ve kilit. Eski nesne korunur; final byte/SHA readback zorunlu. | Yerel sahte servis/crash testleri. Canli OAuth, internet aktarim hizi, kota ve gercek Drive readback bu tur yapilmadi. |
| Mail | Baslangic ve heartbeat maili yok. Yalniz yeni buyuk milestone, TR/ID donusu gerektiren handoff ve terminal durum; kalici outbox. SMTP GPU kritik yolunda degil. | Milestone'lar Pod birakildiktan sonra bounded drain ile toplu gelebilir; her adimin aninda mail teslimi vaat edilmiyor. Belirsiz SMTP sonucunu tekrar tekrar gonderme yok. |

### Yerel dogrulama ve kalan harici kabul

- Son tam test: **1250 passed, 13 skipped, 163,62 saniye**. Atlanan testler PASS sayilmadi. Bu test basladiktan sonra eklenen tek son koruma (worker ACK'i yerel tamamlanma sanip yeniden GPU dongusune girme) ayrica **50 controller testi, 2,55 saniye** ile dogrulandi; bu sayilar birbirine eklenmez.
- Ara tam testler: eski-plan paketi **1103 passed, 12 skipped, 110,18 saniye**; ilk-saat entegrasyonu **1208 passed, 13 skipped, 175,75 saniye**.
- Odakli controller/parca aktarimi, exact current-part hata kaniti, eski lease cleanup, butce, TR/ID handoff ve full-ASR'den once child dispatch testleri gecti.
- Yerel `doctor`: Python, FFmpeg, FFprobe, Git ve rclone OK; torch MISSING. Yerel ASR icin CPU fallback acilmadi; QSV yakma torch gerektirmez.
- Docker tarifi ve RunPod bootstrap/start/worker betikleri incelendi; shell syntax ve diff whitespace kontrolu gecti. Docker build/CI yapilmadi.
- Mevcut kullanici degisiklikleri korundu. Commit/push, yeni Episode 13/14 cevirisi, gercek encode veya canli Drive islemi yapilmadi. Parca kesimi/FFmpeg aralik ve sample davranisi `find-docs` ile birincil [FFmpeg belgelerine](https://ffmpeg.org/ffmpeg-all.html) gore uygulandi; sentetik yerel medya testleri gercek bolum provasi yerine gecmez.
- Kalan harici kabul: immutable image build/digest qualification, tek yetkili gercek EP14 provasi, gercek ses/video kalitesi ve canli Drive readback/sure olcumu. Kullanici ayrica prova istemeden ucretli compute baslatilmaz.

Hedef **14400 saniye**, en fazla **21600 saniye**; bekleme, yeniden baslatma ve parcalar ayni orijinal bolum saatini kullanir. Bu butce bitmeden eksik veya hatali isi PASS saymak yoktur. 4/6 saat gercek teslim basarisi henuz KANITLANMADI.

## Ikinci paket (tarihsel ara kayit)

Asagidaki ilk-paket kaydi tarihsel kanittir. Yeni kullanici talebi Episode 14 icindir; Episode 13 artifactleri degistirilmiyor. Gercek bolum provasi ve ucretli RunPod icin yeni operator talebi yoktur.

- Kod-duzeltme retry yetkisi gercek, atlanmamis pytest fixture sonucu; ayni retained hata; model/ses/kaynak kimligi; degismeyen onceki checkpointler; yalniz ilgili hizalama component'i ve orijinal bolum saatiyle baglandi. `require_resume`, raw ASR ve ses incelemesinde yeni inference yapilmasini engeller.
- Residual/radius/edge/partition adaylari, deterministik retler ve component secimleri hash-bagli checkpointlerden kullanilabiliyor. Kurtarma icin component zaman/deneme tavanlari var. Bunlar sureci kilitleyen bir GPU cagrisi icin hard preemption degildir; dis worker watchdog'u gereklidir.
- Production ceviri politikasi schema/pack kimligine baglandi. Uc context-aware workspace, tekil UID sahipligi, riskli cue'ya ozel anlam/register/dini ifade ve sahne sahipligi inceleme kaydi eklendi. Yeni policy-bound ID ZIP'i, kendisine bagli `.workspace.json` kaniti olmadan production preflight/finalize gecemez. Serbest anlam ve dogallik kontrolu bir reviewer kararidir; basit kuralli test otomatik semantik ispat diye sunulmaz.
- Diagnostic transferleri manifest/SHA ile tekrar kullaniliyor. Degismeyen handoff dosyalari tekrar yuklenmiyor.
- Yeni `local-qsv-v1` yolunda worker strict altyazi exportundan sonra 24 ile doner. Controller disaridan GPU'nun yok oldugunu dogrular; yerel QSV yakma ve Drive yayini bundan sonra calisir. Yerel encode tekrarinda GPU acilmaz. Eski `remote-nvenc-v1` acik secenek olarak kalir.
- E-postalar kalici outbox'a yazilir. Progress olaylari SMTP cagrisi yapmaz. Terminal/action olaylari GPU release sonrasinda bounded drain ile gonderilir; belirsiz SMTP sonucu otomatik tekrar edilmez. Mutable outbox, strict subtitle manifestine dahil degildir.
- Yerel Intel QSV sentetik tek-kare cihaz testi PASS: FFmpeg `n9.0.1-26-g5c8e7e2433-20260905`, Intel surucu `32.0.101.7088`. Bu, bolum encode hizinin, altyazi gorunurlugunun veya perceptual kalitenin kaniti degildir. Bolum medyasi acilmadi.
- Ara tam-suite sonucu: **1031 passed, 12 skipped, 116,63 saniye**. Sonraki degisiklikler icin son tam test ayrica kaydedilecek.

Bu ara kayittan sonra eksik-byte Drive session resume, erken cue baslangici ve gercek ilk-saat strict-partial worker/controller akisi uygulandi; guncel durum yukaridadir.

Kabul siniri: hedef 14.400 saniye, ust butce 21.600 saniye. Gercek bolum kosusu olmadan 4/6 saat teslimi veya Episode 14 canli uretim hazirligi kanitlanmis sayilmaz. Nitelendirilmis immutable image, canli ozel OAuth/Drive readback ve gercek bolum kalite/sure provasi harici dogrulama gerektirir.

## Ilk paket (tarihsel kayit)

Kapsam: Mevcut 12 Eylul plani uzerinde kod ve yerel testler. Zaman/maliyet denetimi bastan yapilmadi. Onceki kirli dokumanlar ve Episode 13 teslim artifactleri korundu. Commit, push, yeni ceviri, encode, Drive yayini veya ucretli RunPod yapilmadi.

## Uygulanan duzeltmeler

| Alan | Degisiklik | Kanit siniri |
|---|---|---|
| Ayni hatayi tekrar calistirma | Hash-bagli terminal hata kaydi, provider olusturmadan once degismeyen girdiyi engeller. Yalniz kod/encoder parametresi degisikligi yeni GPU denemesi acmaz. Yerelde dogrulanmis TR/ID donusu degisirse en fazla 2 ek deneme. | Yerel sahte provider testleri. Eski baglanmamis hata kaydi fail-closed kalir. |
| Bolum saati | Handoff beklemesi artik saatten dusulmez. En erken bolum baslangici korunur; hedef 14.400 saniye, varsayilan sert ust sinir 21.600 saniye. Drive'a kalan sure aktarilir; daha buyuk ortam degiskeni reddedilir. | Butce testleri. Her asamanin planlanan kilometre tasinin uygulanmasi ve 4 saatlik teslim kaniti henuz yok. |
| Calisma ortami | Controller, bootstrap ve worker Python yolu `/opt/venv` oldu. Digest ile sabitlenmis image zorunlu; provider `imageName` eslesmesi kontrol edilir. Immutable modda ABI/paket farki kurulumla kapatilmaz. Image'a ABI marker ve anahtarli SSH baslangici eklendi. | Yerel test ve shell syntax kontrolu. Docker build yapilmadi, nitelendirilmis image digest mevcut oldugu iddia edilmiyor. `imageName` eslesmesi provider'in gercekte yukledigi digest icin bagimsiz attestation degildir. |
| Hizalama tekrar maliyeti | Cozulmus cakisma yeniden denenmez. Dogrulanmis kenar adayi sonrasi gereksiz partition denemeleri durur. Baslangictaki basarili cakisma grubu secimleri ilgili cue, ham CTC sonucu, VAD, model/kod/politika kimligiyle yeniden kullanilir. | Cold/cache ciktisi ve hash esitligi; yerel invalidation ve bozuk cache testleri. Son akustik validatorlar korunur. |
| Sahneye ait kelime | Tum bolum metni ayni olsa bile her cue'nun kelimeleri kendi akustik araliginda olmali. Gosterilen `primary_text` ayni cue'nun `timing_text` metniyle uyusmali. Kaynakta olmayan konusmaciya eklenen replik reddedilir. | 30 saniye aralikli elma/armut fixture'i; kayip sahne, erken cue sonu ve farkli bilinen konusmaci overlap testleri. |
| Okunabilirlik | Yeni strict uretim icin production satir limiti 42 karakter ve 2 satir. Turkce bolme bunu ceviri paketi dondurulmadan once uygular. 7 saniyeyi asan cue hard QA hatasidir. | Metin kesilmez, kelimeye yapay zaman eklenmez, sonraki sahne kaydirilmaz. Eski emergency artifactler yeniden yazilmadi. |
| Ceviri kalitesi | ID return asamasina mevcut dini ifade, Allah, isim, sayi ve okuma hizi kontrolleri tasindi; final validator da aynen kaldi. Bekleyen review final uretime gecemez. | Yerel kuralli QA. Dogal Endonezce ve serbest anlam esdegerligi tum bolum icin otomatik kanitlanmis degildir. |
| Drive tekrarlari | Kalici hash-bagli upload kaydi ve tamamlanmis partial nesne yeniden kullanimi. Metadata/readback hatasi tum dosyayi otomatik bastan gondermez. Nesne kimligi ve tam byte/SHA yeniden dogrulanir; upload baslatma en fazla 2 deneme. | Yari kalmis upload oturumunun byte seviyesinde devam ettirilmesi degildir. Gercek aktarim hizi olculmedi. |
| Drive on kontrolu | Ozel OAuth istemcisi/refresh kimligi, yazma kapsami ve bos alan kontrolu. Compute oncesi kimlik/readiness, yayin oncesi kesin dosya boyutu kontrolu. Receipt'te yalniz kimlik hashleri saklanir. | Canli OAuth, yazma yetkisi veya API kota garantisi yok. Eski nesne korunmasi ve tam final readback zorunlulugu devam eder. |

## Episode 13 icin durust kalite durumu

- Kullanicinin gordugu gercek sahnelerin zaman damgalari verilmedi; ses/video incelemesi yapilmadi. O sahnelerdeki kelime atamasinin kesin ASR/ceviri kok nedeni UNKNOWN.
- Somut kod acigi yeniden uretildi: eski segmentation validatoru butun metnin korunmasina bakiyor, metnin dogru cue/zamana ait oldugunu kontrol etmiyordu. Bu acik ve bagimsiz incelemede bulunan iki ek gecis kapandi.
- Uzun sessizlikten bolme ve strict VAD/eksik konusma kontrolleri zaten vardi. Bunlar bu tur yeni gelistirilmis gibi sayilmadi; katilastirilmis cue kontrolleriyle birlikte test edildi.
- Episode 13'te yayinlanan segment-timed emergency yol, strict akustik kabul yerine gecmez. Eldeki SRT/MP4 ve Drive dosyasi degismedi; `REVIEW_REQUIRED` durumu suruyor.

## Ilk paket sonunda kaydedilen uygulama sirasi (tarihsel)

1. Kod duzeltmesini retained hata fixture'i ve yalniz etkilenen component resume planiyla baglayan acik retry yetkisi. Simdiki koruma kod-only ve eski baglanmamis hatalarda guvenli bicimde durur; bu yeni yetki akisinin yerine gecmez.
2. Sonraki residual/radius/edge/partition aramalarinin kalici component checkpoint'i. Bu pakette yalniz basarili ilk conflict gruplari cache'lendi.
3. Ortak production politika kimligini ve exact ceviri pack freeze bagini tum girislere tasimak; 3 baglamli paralel ceviri parcasi ve risk-odakli anlam/register QA akisi. Yeni raw-MT veya tam bolum yeniden ceviri bu tur yapilmadi.
4. Diagnostic manifest/hash dedup ve degismeyen handoff upload'larini atlama.
5. Dogrulanmis subtitle toplama sinirinda GPU birakma, versioned yerel QSV encode yolu ve encode oncesi kesin disk/capability kontrolleri. Mevcut worker final encode ayrimi bu tur degismedi.
6. Gercek eksik upload-session resume destegi; ozel OAuth ve alan/aktarim staging dogrulamasi. Eski teslimi koruma ve tam SHA readback kaldirilmadan.
7. Kalici mail outbox ve GPU kapandiktan sonra bounded drain. Mevcut senkron mail bu tur degismedi.
8. Kullanici ayrica gercek bolum provasini istediginde image build/digest qualification, tek bounded GPU kosusu, encode ve canli Drive readback ile 240 dakika hedef / 360 dakika ust sinir olcumu.

## Dogrulama

- Son yerel tam test: `.venv/Scripts/python.exe -m pytest -q tests`, **973 passed, 12 skipped, 95,28 saniye**. Atlanan testler PASS sayilmadi.
- Odakli testler her alt pakette, sonra bagimsiz incelemede bulunan iki ek cue/konusmaci acigi kapatildiktan sonra tam test yeniden calistirildi.
- `git diff --check`: exit 0. Git mevcut Windows satir-sonu donusum uyarilarini gosteriyor; whitespace hatasi yok.
- Bootstrap/container-start/run-episode shell syntax kontrolleri gecti. Docker build, CI, gercek GPU, ses dinleme ve canli Drive dogrulamasi yapilmadi.

`find-docs` ile RunPod custom image SSH baslangici ve rclone retry/OAuth davranisi guncel birincil belgelerden kontrol edildi: [RunPod custom template](https://docs.runpod.io/pods/templates/create-custom-template), [rclone retries](https://rclone.org/docs/#retries-int), [Drive OAuth](https://rclone.org/drive/#making-your-own-client-id). Bu belge okumasi canli hesap/islem dogrulamasi degildir.

4 saat hedefi SAGLANDI denmiyor. 6 saat siniri teslim garantisi degil, eksik sonucu PASS yapmadan durdurma politikasidir.
