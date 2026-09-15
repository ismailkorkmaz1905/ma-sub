# Episode 13 zaman ve maliyet kaniti denetimi

Kesim: 12 Eylul 2026 12:45:03 SGT. Baslangic: 11 Eylul 2026 06:00:00 SGT.
Toplam takvim suresi 30 saat 45 dakika 3 saniye. Bunun tamamini GPU suresi veya
bosa harcanan sure olarak adlandirmak kanitlarla desteklenmiyor.

## Ana bulgu

- 92 CLI logu var. 89 tanesinde `run_finished` var; 3 tanesinin bitisi UNKNOWN.
- Tamamlanan 89 CLI araliginin kesisimsiz birlesimi 56,621.303 saniye, yani
  15 saat 43 dakika 41.303 saniye. Aralarinda cakisma yok.
- 23 benzersiz forced-alignment calismasi FAIL ile bitmis: 22,765.7 saniye,
  yani 6 saat 19 dakika 25.7 saniye. Ek 5 benzersiz hizalama calismasinda
  terminal sonuc yok; heartbeat kayitlari en az 4,867.5 saniye daha gosteriyor.
  Boylece hizalamada gorulen toplam alt sinir 27,633.2 saniye, yani
  7 saat 40 dakika 33.2 saniye. Bu GPU fatura suresi degildir.
- Onceki rapor ayni remote logun tekrar okunmasini yeni asama calismasi saymis.
  Ilk 6 asamada 18 terminal satiri ve 470.3 saniye cift sayilmisti.
  Benzersiz terminal asama toplami 31,251.6 saniye, yani 8 saat 40 dakika
  51.6 saniye. Biten ve bitmeyen CLI loglarini birlikte kapsar.

## Takvim suresinin kesisimsiz ayrimi

| Kalem | Sure | Anlami |
|---|---:|---|
| 06:00:00 ile ilk CLI baslangici 06:31:19.342 arasinda | 1,879.342 saniye | CLI kaniti yok; faaliyet UNKNOWN |
| 89 tamamlanmis CLI araligi | 56,621.303 saniye | Baslangic/bitis UTC damgalarinin birlesimi |
| Ilk ve son CLI arasinda bu 89 araligin disi | 44,399.154 saniye | 12 saat 19 dakika 59.154 saniye; bitissiz loglar ve diger faaliyetler dahil, idle degil |
| Son CLI bitisi 10:34:59.799 ile kesim 12:45:03.001 arasinda | 7,803.203 saniye | Kurtarma, ceviri, encode, aktarim ve inceleme burada olabilir; tam faaliyet paylastirmasi UNKNOWN |
| Toplam | 110,703.001 saniye | 30 saat 45 dakika 3.001 saniye |

## On bir adim: tekrar log satirlari ayiklanmis sayim

Sure sutunlari yalniz terminal PASS/FAIL zamanlayicilaridir. Medyan ve maksimum
ayni terminal kumesinden hesaplanir. Baslayan fakat bitisi gorulmeyen hizalamalar
ayri UNKNOWN sutunundadir. Son dort adimdaki sifir, bu strict CLI loglarinda
terminal olay olmadigi anlamindadir; ayri kurtarma calismasinin yapilmadigi
anlamina gelmez. Mevcut kod `strict_finalize` ve `burn_mp4` adimlarini ayirir;
onceki on bir adimlik raporla karsilastirma icin burada ayni satirdadir.

| Adim | Benzersiz baslama | PASS | FAIL | Terminal UNKNOWN | Terminal toplam (saniye) | Medyan (saniye) | Maksimum (saniye) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. download | 37 | 33 | 4 | 0 | 657.9 | 14.8 | 93.5 |
| 2. audio | 33 | 33 | 0 | 0 | 167.4 | 3.5 | 43.9 |
| 3. raw_asr | 30 | 30 | 0 | 0 | 5,071.2 | 89.25 | 2,407.9 |
| 4. tr_pack | 30 | 30 | 0 | 0 | 647.1 | 21.05 | 28.4 |
| 5. tr_return | 29 | 29 | 0 | 0 | 71.5 | 2.4 | 3.4 |
| 6. audio_review | 29 | 28 | 1 | 0 | 1,870.8 | 11.1 | 623.5 |
| 7. forced_alignment | 28 | 0 | 23 | 5 | 22,765.7 | 972.7 | 2,790.8 |
| 8. id_pack | 0 | 0 | 0 | 0 | 0 | UNKNOWN | UNKNOWN |
| 9. id_return | 0 | 0 | 0 | 0 | 0 | UNKNOWN | UNKNOWN |
| 10. finalize / strict_finalize / burn_mp4 | 0 | 0 | 0 | 0 | 0 | UNKNOWN | UNKNOWN |
| 11. drive_readback | 0 | 0 | 0 | 0 | 0 | UNKNOWN | UNKNOWN |

Ham terminal satiri sayilari onceki raporla ayni: 40, 36, 33, 33, 32, 32, 23.
Fakat `START` sonrasindaki ayni EMAIL `message_id` ayni asama calismasini tekrar
gosterir. Ornegin `run-20260911T072025.574392Z-11656.log` ile
`run-20260911T073906.429499Z-14484.log` ayni ilk 6 asamanin 144.2 saniyesini
tekrar gosterir; ikincisinde `start response lost` ve `remote job status=LOST`
vardir. `src/mas/runpod_controller.py` remote monitor baslangicinda `offset = 0`
kullanir. `src/mas/notify.py` her gercek mesaj icin `make_msgid()` uretir.
Bu nedenle ayni mesaj kimligi yeniden calisma kaniti degildir.

Bes terminali eksik hizalamanin son heartbeat sureleri 62.2, 2,461.7, 1,471.3,
811.2 ve 61.1 saniyedir. Her birinin mesaj kimligi ve kaynak logu JSON'da
`unique_unfinished` altindadir. Son heartbeat terminal suresi degil, alt sinirdir.

## Tekrar basarisizliklari ve diger maliyetli isler

Asagidaki CLI sureleri asama tablosuna EKLENMEZ. Her satir run icindeki ilk
`CAUSE:` satirina gore gruplanan butun CLI sureleridir; kurulum, islem, toplama
ve kapanis birlikte kapsanir. Ayni kok nedenin her ayrintisinin degismedigini
veya bu surenin tamaminin onlenebilir oldugunu iddia etmez.

| Gozlenen hata grubu | CLI sayisi | CLI elapsed toplami (saniye) |
|---|---:|---:|
| Ayni `2879ed635be09328` / `c7d307c5163105c2` komsu UID cakismasi | 7 | 12,835.219 |
| Ayni `2c9ca86ab31b1235` / `e813b3f56cb23a94` komsu UID cakismasi | 6 | 10,505.781 |
| Ayni `e7646357031edfa8` / `2879ed635be09328` komsu UID cakismasi | 2 | 3,732.220 |
| Remote pipeline exit 124 | 2 | 3,296.422 |
| Capacity ownership belirsiz | 9 | 1,959.485 |
| Network command exit 124 | 4 | 1,858.563 |

- Bootstrap: loglarda 46 adet `10 upgraded, 126 newly installed` satiri var.
  Onceki rapordaki "136 newly installed" dogru degil: 126 yeni paket ve 10
  guncelleme. Ham 80 `Fetched ... MB in ...s` satiri 589 saniye ve 4,661.7 MB
  gosteriyor; bu tam bootstrap suresi veya tekillestirilmis ag aktarimi degil.
  Tam kurulum, model yukleme ve hazirlik icin ayri sure UNKNOWN.
- Kaynak hash kontrolu: 38 `local source seed verified` satiri var. Her biri
  1,076,347,555 byte kaynagi gosteriyor. 38 x boyut = 40,901,207,090 byte
  yerel dogrulama girdisi. Bunun iki kati olan 81.8 GB'yi yalniz bu satirlardan
  gercek yerel+remote hash okuma toplami diye kabul etmek mumkun degil.
- Hata kaniti toplama: 150 `diagnostic downloaded` satiri var. Her aktarimin
  sure damgasi olmadigindan toplam indirme suresi UNKNOWN. Onceki rapordaki
  1.849 GB toplam / 1.702 GB tekrar sayisi bu denetimde bagimsiz hesaplanmadi.
- E-posta: 521 `sent` satiri, 482 benzersiz mesaj kimligi var. 39 tekrar log
  gorunumunu yeni SMTP gonderimi saymamak gerekir. `send_email` senkron calisir.
  Baslangic e-postasi asama zamanlayicisina dahildir, sonuc e-postasi disindadir;
  tum mail gecikmesini ayrica toplamak cift sayim yapar. SMTP toplam suresi
  UNKNOWN; Message-ID uretilme zamani gonderim tamamlanma zamani degildir.
- En buyuk tamamlanmis-CLI-disi araliklar: 11 Eylul 12:06:45-15:19:23 SGT
  11,558.029 saniye; 20:23:34-22:50:37 SGT 8,822.830 saniye. Ikinci aralikta
  bitisi olmayan bir bootstrap logu vardir. Bu araliklari tamamen bekleme,
  ceviri veya kod yazma suresi olarak etiketlemek desteklenmiyor.

## Onceki rapora gerekli duzeltmeler

`docs/EP13_RUNTIME_AUDIT_2026-09-12.md` icin:

1. 06:20 yerine kullanicinin istedigi 06:00 kullanilmali; ilk CLI 06:31:19.
2. 89 tamamlanmis log ve 3 bitissiz log ayri yazilmali.
3. 56,625.172 saniye monotonic `elapsed_seconds` toplamidir. UTC baslangic/bitis
   birlesimi 56,621.303 saniyedir. 3.869 saniye fark iki olcme sinirindan gelir;
   kesisim oldugu anlamina gelmez.
4. 31,721.9 saniye ham terminal toplami 470.3 saniye replay icerir; benzersiz
   terminal toplam 31,251.6 saniyedir. Hizalama terminal medyani 972.7 saniye,
   yani 16.212 dakika; 16.4 dakika degil.
5. Terminali eksik 5 benzersiz hizalama calismasini atlamak, en az 4,867.5
   saniyeyi kaybettirir. Bunlarin gercek sonuclari UNKNOWN.
6. Ham stage toplamindan CLI toplam cikarilarak elde edilen "controller
   overhead" kesin altyapi suresi degildir. Replay, terminali eksik asamalar,
   bitisi eksik CLI'lar ve e-posta sinirlari once ayrilmalidir.
7. 30 saniyelik QSV orneginden uretilen 36.4 dakika tam-bolum suresi bir
   ekstrapolasyondur. Gercek encode receipt suresi varsa onunla ayri yazilmali.

## Strict CLI disindaki gercek kurtarma islemleri

Kaynak klasoru: `EPISODES/Muhtemel Ask 13.Bolum/emergency/segment-timing/`.
Bu islemler yukaridaki strict stage sayaclarina girmemis; sifir sayaci is
yapilmadigi anlamina gelmez. Asagidaki sureleri takvim toplamindan sonra
yeniden eklememek gerekir.

| Is | Olcum | Kanit siniri |
|---|---:|---|
| Tam bolum, QSV ile altyazi yakma | 1,940.25 saniye, yani 32 dakika 20.25 saniye | `Muhtemel Ask 13.Bolum - SEGMENT-TIMED REVIEW.burn.json`, `encode_elapsed_seconds` |
| Uretilen burned MP4 | 5,548,032,992 byte | `drive_review_receipt.json`, `local_burned_review.bytes`; bu byte sayisi bu kesimdeki Drive nesnesine ait degil |
| Kucuk soft-sub MP4 remux | 14.16 saniye | Ana gorevin komut suresi gozlemi; bu alt gorev bagimsiz zaman dosyasi bulmadi |
| Onceki kucuk MP4 upload | 12:25:35-12:27:48 SGT, yaklasik 133 saniye | Ana gorevin gozlemi; receipt transfer baslangic/bitis suresini saklamiyor |
| Kucuk MP4 tam readback | 1,076,298,665 byte, PASS | `drive_review_receipt.json`, `files[0]`; yerel receipt incelemesi, bu alt gorev yeni canli Drive okuması yapmadi |

Receipt `completed_at=2026-09-12T04:36:41.5280484Z`, yani 12:36:41.528 SGT.
Kucuk video ve iki SRT icin review delivery kaydi var; `strict_eligible=false`.
Receipt'teki buyuk video icin upload atlama gerekcesi o andaki eski durumdur.
Ana gorevden iletilen yeni HQ upload devam ediyor; bu rapor onun bitisine veya
canli remote byte/SHA sonucuna PASS vermez. Ceviri, kalite kontrol ve sonradan
dosya degistirme calismalarinin ayri sureleri UNKNOWN.

## Harcama siniri

### Kod nerelerde degisti?

Ana gorevin yerel Git denetimi, `main` uzerinde 11 Eylul 2026 06:00 SGT
sonrasinda 53 commit gosterir; sonuncusu `c454a90` (12 Eylul 12:31:27 SGT).
Bu 53 commit, 53 push, 53 GPU kosusu veya 53 basarili duzeltme demek degildir.

| Dosya | Dosyaya dokunan commit sayisi | Degisiklik alani |
|---|---:|---|
| `src/mas/engine/forced_align.py` | 25 commit | Hizalama, cakisma adaylari ve recovery |
| `src/mas/runpod_controller.py` | 14 commit | Uzak is, resume, kapasite ve izleme |
| `runpod/bootstrap.sh` | 6 commit | Worker ortam kurulumu ve reuse |
| `src/mas/pipeline.py` | 5 commit | Asama ve checkpoint koordinasyonu |
| `src/mas/engine/audio_review.py` | 4 commit | Akustik review |
| `src/mas/emergency_segment.py` | 2 commit | Ayri segment-timed kurtarma yolu |

Sayim `git log --since='2026-09-11T06:00:00+08:00' --name-only --format=`
ciktisinda dosya gorunumlerini gruplar. Satirlar birbiriyle toplanmaz; bir
commit birden fazla dosyaya dokunabilir. Commit frekansi tek basina zaman veya
kalite olcumu degildir. Son uc commit kurtarma yolu (`a1bedba`), return binding
ve dini ifade dogrulamasi (`968836a`), tek-poll izleme (`c454a90`) kapsamindadir.
Son kullanici isteginden sonra uretim kodu degistirilmemistir; yeni belgeler
commit/push yapilmadan yerelde tutulmustur.

### Ucret kanitlari

Bu denetim yeni ucretli islem, API model cagrisi veya RunPod baslatma yapmadi.
CLI duvar suresi, ASR/CTC zamanlayicisi ve provider GPU fatura suresi farklidir.

Ana gorevden iletilen provider gozlemleri: ilk bakiye 2.302372591 USD; pozitif
bakiye sicramalari toplami 9.8863361889 USD; son bakiye 4.4549433716 USD.
Bu alt gorev provider kayitlarini bagimsiz yeniden okumadi. Yalniz tum para
girislerinin bu sicramalar oldugu varsayilirsa cebirsel tuketim
`2.302372591 + 9.8863361889 - 4.4549433716 = 7.7337654083 USD` olur.
Bu varsayim dogrulanmadigi icin sonuc FATURA DEGILDIR. Ornekler arasindaki
harcama, iadeler, diger kaynaklar ve eksik bakiye hareketleri ayrismiyor.
Kesin GPU, depolama ve model/API dolar tutarlari UNKNOWN.

Ana gorevin canli Codex aracindan ilettigi hesap geneli haftalik limit
kullanimi yuzde 92; pencere 10,080 dakika; sifirlama 15 Eylul 2026
17:29:13 SGT. Bu Episode 13 token sayisi veya dolar faturasi degildir.
Global oturumlarin mesaj/icerikleri taranmadi. Tekrarlanan ve miras alinan
oturum token sayaclarini toplayarak yeni bir harcama tutari uretilmedi.

## Yeniden hesaplama ve kapsam

### HQ teslim eki

Bu ek 12:45:03 kesimindeki CLI/asama toplamlarini degistirmez. Ayri HQ
teslimi 12 Eylul 2026 13:37:04.706 SGT'de tamamlanmistir.

- HQ aktarim sureci 12:42:59 SGT'de basladi; 13:27:30 SGT'de exit code 0
  verdi: yaklasik 44 dakika 31 saniye. Ilerleme toplami bir, iki, uc dosya
  boyutuna cikti (5.167, 10.334, 15.501 GiB); yuzde 100 satirlari tek basina
  teslim kaniti degildi. Iki yeniden aktarim goruldu.
- Ayni donemde 13:07 ve 13:17 SGT metadata/kota sorgulari ortak Google API
  istemcisinin `RATE_LIMIT_EXCEEDED` 403 hatasini verdi. Bu, depolama alani
  hatasindan farklidir. 44 dakika 31 saniyenin tamaminin kota beklemesi oldugu
  olculmedi.
- Tam readback 13:28:31 SGT'de basladi; byte/SHA sonucu 13:33:44 SGT'de
  goruldu: yaklasik 5 dakika 13 saniye, yerel hash kontrolu dahil.
- 5,548,032,992 byte ve SHA-256
  `f2043cdc32b4ce6911d6310d718beb1ae60f30cd7930c62d950d5e12158c6f7d`
  yerel HQ ile ayni. Final Drive ID `1UEaQ_BNV9vPLn1EJyliwK3et0wOWFtFL`.
- Eski kucuk MP4 `1BIer73iRXKOCNwcUt7mVIrnIObPTmtNf` Drive cop kutusuna
  tasindi; kalici silinmedi. Iki SRT korundu. Gecici yerel readback kopyasi
  temizlendi; asil yerel HQ korundu.
- Baslangictan replacement receipt zamanina kadar yaklasik 54 dakika
  5 saniye gecti. Bu gozlem plandaki 25 dakikalik Drive hedefini KARSILAMAZ.
  Ortak kota ve tum dosyayi yeniden aktarma davranisi giderilmeden bu hedef
  production icin dogrulanmis sayilamaz.
- Kanit: Episode 13 emergency/segment-timing altindaki
  `drive_hq_replacement_receipt.json` ve Luna alt gorevinin zaman damgali
  aktarim ciktilari. Root final Drive ID/ad/boyutunu ayrica connector ile
  okudu. Tarayicidan ikinci bir upload yapilmadi.
- Bu `VERIFIED_REVIEW_DELIVERY_HQ` sonucudur; `strict_eligible=false` ve
  altyazi akustik kabul durumu `REVIEW_REQUIRED` olarak kalir.

### Denetim yontemi

- Betik: `var/ep13_timeline_audit.py`; veri: `var/ep13_timeline_audit.json`.
  Ikisi de ignore edilen audit kanitidir. Calistirma:
  `.venv\Scripts\python.exe var\ep13_timeline_audit.py 2026-09-12T04:45:03.001458+00:00`.
- Yalniz Episode 13 `logs/run-*.log` satirlari okunur. Kaynak medya, WAV,
  altyazi icerigi, credentials veya diger bolumler okunmaz.
- Betik `run_started` / `run_finished` damgalarini kullanir, bitisi eksik
  araliklari birlesime dahil etmez, tamamlanan araliklari birlestirir.
  Stage replay ayiklamasi stage adi + START bildirim `message_id` uzerindendir.
- JSON her terminal olayinin kaynak dosyasini, satirini, mesaj kimligini ve
  suresini korur. Satir numaralari Python universal-newline yorumundadir;
  carriage-return progress satirlari nedeniyle diger araclarla fark olabilir.
- Bu yerel log denetimidir. Yeni test, CI, GPU, Drive veya provider PASS
  iddiasi yoktur. Uretim kodu degistirilmedi.
