# V2 New Episode Prompts

Replace `X` with the episode number.

These prompts belong only to the isolated `SYSTEM_V2_BETA` workflow. They do
not repair a V1 output in place. For an old episode, start a new V2 run from
`01_PREPARE_TR.ipynb`; do not mix V1 packs, output ZIPs or schema hashes into
the run.

## Operator run order

1. Open `SYSTEM_V2_BETA/01_PREPARE_TR.ipynb`, set `EPISODE` and `SOURCE_URL`,
   and run all.
2. Send the Turkish prompt below. Wait for the exact verified output ZIP.
3. Open `SYSTEM_V2_BETA/02_ALIGN_PREPARE_ID.ipynb`, set the same `EPISODE`, and
   run all.
4. Send the Indonesian prompt below. Wait for the exact verified output ZIP.
5. Open `SYSTEM_V2_BETA/03_FINALIZE_V2.ipynb`, set the same `EPISODE`, and run
   all.
6. Review playback. Then open `SYSTEM_V2_BETA/04_ARCHIVE_V2.ipynb` and run its
   dry run before enabling cleanup.

A notebook PASS is not a zero-error guarantee; the pilot still requires human
listening and playback review.

## After 01_PREPARE_TR finishes

```text
Bölüm X için Türkçe düzeltme aşamasını tamamla.

Drive'da şu exact paketi bul:
Muhtemel_Ask_Subtitles/EPISODES/Muhtemel Ask X.Bolum/translation_input/Muhtemel Ask X.Bolum_TR_CORRECTION_PACK.zip

ZIP içindeki TR_CORRECTION_INSTRUCTIONS.md talimatını eksiksiz uygula. Her kaydın immutable alanlarını, asr_audit dahil, birebir koru. Yalnız Türkçeyi düzelt; Endonezceye çevirme. Immutable audio-review kaydı olmayan normal ASR kayıtlarının tamamı non_dialogue=false, audio_reviewed=false ve review_disposition=not_applicable kalmalı; tr_corrected alanına boş olmayan düzeltilmiş Türkçe yazılmalı ve kayıt gürültü diye atılmamalı.

Bu aşama yalnız metin düzeltmesidir. WAV dosyalarını açma, inceleme, dinleme veya transcribe etme. Ses kararlarını sonraki Colab notebooku verecek. Web araması yapma. Hugging Face, GitHub, model deposu veya başka website arama; Whisper ya da başka ASR modeli indirme/kurma; transkripsiyon plugini arama; harici transkripsiyon veya çeviri servisi kullanma. Türkçe düzeltmeyi kendi dil yeteneğinle ve yalnız JSON metin kanıtlarıyla yap.

İzin verilen tek harici connector, yukarıdaki exact giriş ZIP'ini okumak ve exact çıkış ZIP'ini yazmak için Google Drive'dır. Yerel ZIP/JSON işleme serbesttir; ağ erişimi, paket kurulumu ve model/tool keşfi yasaktır.

Exact hash-bound unresolved_vad_speech, suspected_asr_hallucination ve orphan_youtube_caption kayıtlarının tamamında non_dialogue=false, review_required=true, audio_reviewed=false ve review_disposition=pending_audio_review bırak. ASR/YouTube metni bulunan adayın Türkçesini düzelt; ikisi de boş speech hole için tr_corrected alanını boş bırak. audio_reviewed=true, confirmed_dialogue, reviewed_non_dialogue veya discarded_asr_hallucination kullanma. Bağlamdan konuşma ya da sessizlik uydurma.

Noktalama/sadece büyük-küçük harf değişikliği dışında hiçbir lexical ASR tokenini exact audio-review kaydı olmadan silme. Silmek gerekiyorsa review_required=true, audio_reviewed=false ve review_disposition=pending_audio_review ile exact UID'yi bildir; 01_PREPARE_TR notebookunu bu UID `EXTRA_AUDIO_REVIEW_UIDS` içine eklenerek yeniden çalıştırmadan tamamlandı deme. Notebook pack değişikliğinin yalnız ek hash-bound audio-review kanıtı olduğunu doğrularsa mevcut text-only çıktıyı otomatik yeniden bağlar; bu durumda 2.977 kaydı Pro'ya tekrar düzelttirme.

Exact çıktıyı Drive'a yaz ve geri açıp doğrula:
Muhtemel_Ask_Subtitles/EPISODES/Muhtemel Ask X.Bolum/translation_output/Muhtemel Ask X.Bolum_TR_TEXT_CORRECTED.zip

Dosya yazılıp doğrulanmadan tamamlandı deme. Uzun işlemde kısa durum güncellemeleri ver.
Tamamlandığında kullanıcıya sıradaki adımın SYSTEM_V2_BETA içindeki
02_ALIGN_PREPARE_ID.ipynb olduğunu ve aynı EPISODE değerini kullanması
gerektiğini söyle. Final TR_CORRECTED.zip dosyasını Pro değil, bu notebookun
hash-bound Colab ses denetimi oluşturacak.
```

## After 02_ALIGN_PREPARE_ID finishes

```text
Bölüm X için yalnız Endonezce çeviri aşamasını tamamla.

Drive'da şu exact paketi bul:
Muhtemel_Ask_Subtitles/EPISODES/Muhtemel Ask X.Bolum/translation_input/Muhtemel Ask X.Bolum_ID_TRANSLATION_PACK.zip

ZIP içindeki ID_TRANSLATION_INSTRUCTIONS.md talimatını eksiksiz uygula. Yalnız tr_text alanını doğal, konuşma diline uygun Endonezceye çevirip id_final alanına yaz; kelime kelime makine çevirisi yapma. Karakterlerin ilişkisini, resmiyet seviyesini, romantizmi, öfkeyi, alayı, mizahı, hakareti, tereddüdü ve yarım kalan konuşmayı koru. Normal samimi diyalogda aku, kamu, nggak, udah ve aja kullanılabilir; resmi veya saygılı sahnelerde bağlama göre saya, Anda, Pak ve Bu kullan. Argo zorlaması yapma, anlamı sansürleme/yumuşatma, açıklama veya konuşmacı etiketi ekleme, replikleri kayıtlar arasında taşıma. Altyazıyı kısa ve doğal tut. glossary.json içindeki canonical isimleri, tekrarlar dahil bütün sayı/para değerlerini ve dini ifade karşılıklarını eksiksiz koru.

Çeviriyi kendin yap. Web araması, online çeviri, harici servis, model indirme veya çeviri plugini kullanma.

block_uid, block_index, timing, tr_text, alignment_provenance, schema_version ve schema_sha256 dahil bütün input alanlarını birebir koru. Kayıt bölme, birleştirme, atlama, ekleme veya sıralama yapma.

Exact çıktıyı Drive'a yaz ve geri açıp doğrula:
Muhtemel_Ask_Subtitles/EPISODES/Muhtemel Ask X.Bolum/translation_output/Muhtemel Ask X.Bolum_ID_TRANSLATED.zip

Dosya yazılıp doğrulanmadan tamamlandı deme. Uzun işlemde kısa durum güncellemeleri ver.
Tamamlandığında kullanıcıya sıradaki adımın SYSTEM_V2_BETA içindeki
03_FINALIZE_V2.ipynb olduğunu ve aynı EPISODE değerini kullanması gerektiğini
söyle. Çeviri ZIP'inin hazırlanmasını final yayın veya sıfır hata garantisi
olarak sunma.
```

## After 03_FINALIZE_V2 passes

The final V2 output is under `final/`; `source/` is still the resumable input.
The source video and final MKV coexist temporarily for stream-copy verification.
After human playback review, run `04_ARCHIVE_V2.ipynb` with
`DELETE_INTERMEDIATES=False` first. Verified cleanup then leaves exactly:

```text
final/Muhtemel Ask X.Bolum.mkv
final/subtitles/Muhtemel Ask X.Bolum-id.srt
final/subtitles/Muhtemel Ask X.Bolum-tr.srt
```
