# Episode 11 five-clip pilot result

Status: REVIEW_REQUIRED, not accepted for Episode 12. Real CUDA ASR and local SRT generation completed. Perceptual listening was NOT_PERFORMED: this session has no tool that supplies the local audio as a listenable model input. Model timing comparisons and file generation are not acoustic PASS.

## Reviewable output

- Final local candidate: `EPISODES/Muhtemel Ask 11.Bolum/work/subtitle-pilot/contextual-asr-20260906-4/`.
- `Episode11-five-clips-pilot.srt`: 54 cues covering five source windows totaling 155 seconds, not the full episode. SHA-256 `69a813d25e705c6e1692659b4f9266080040ae026127a371c0d715cffabd0963`.
- `review.html`: original WAV playback, waveform, cue display and seek buttons. Each sample also has `clip-relative.srt`, source-relative `draft.srt`, checksum-bound evidence and assessment JSON.
- Original raw ASR: `var/disposable-asr-pilot-20260906T072407Z-7032e0c9/asr/raw/`. ASR result checksum `955cd1aa55756cf3f115231805dee2574ee8ccf0995f8f50a19d3a81a0dbb4ec`.
- Diarization remains separate under `var/native-diarization/nemo-0.1.0/contextual-five-20260906/`. Raw 80-millisecond probability frames were joined by verified source/clip/manifest hashes. Candidate speaker labels are not established person identities.

## Measured quality limitations

| Clip | Source start, seconds | Cues | Cues shorter than 700 milliseconds | Cues above 20 characters/second | Words with uncertain speaker |
|---|---:|---:|---:|---:|---:|
| low-score-context | 60 | 9 | 1 | 1 | 10 |
| edited-score-context | 350 | 10 | 0 | 5 | 8 |
| drift-duration-context | 470 | 12 | 0 | 0 | 9 |
| unclassified-control | 600 | 13 | 1 | 2 | 8 |
| final-conflict-context | 4280 | 10 | 3 | 4 | 6 |
| Total | - | 54 | 5 | 12 | 41 |

Source: final candidate `assessment.json`. Thirty cues retain unknown aggregate speaker identity. There are no line-length or unresolved-overlap warnings in this candidate. No native frame has two speaker probabilities at least 0.5; this does not prove that the source contains no overlapping speech.

The first ASR cue in low-score-context starts 860 milliseconds before the overlapping diarization speech onset and ends 120 milliseconds before its corresponding offset. The first edited-score-context cue starts 580 milliseconds before the detector onset and ends 380 milliseconds before its offset. These are disagreements between two models on the same exact PCM, not measured human judgments of early/late subtitles. The assessment retains each cue's comparison and its limits. Interior cue boundaries need not coincide with full speaker-turn boundaries.

Short speech at clip edges and uncertain speaker changes remain visible. Missing dialogue, lexical accuracy, real speaker identity and perceptual start/end synchronization are not verified. Do not silently stretch timestamps, rewrite ASR text, erase uncertainty or label these checks PASS to begin Episode 12.

## Concrete repair and validation

The initial join made 90 cues because every unknown-to-candidate speaker transition split a phrase. `speaker_groups` now splits only at a change between supported different speaker labels. Any uncertain word keeps the aggregate speaker unknown, and the original word-level issue remains. A regression verifies that A -> unknown -> B still separates A and B. The local regrouper also balances a short trailing fragment within the same already-separated group without changing words or their timestamps.

Local focused tests: 26 passed in 0.31 seconds. Full local suite with FFmpeg: 706 passed, 5 skipped in 58.93 seconds. These are local results, not CI or acoustic approval. `git diff --check` passed. Docker definition and RunPod scripts were inspected; no new image build is claimed.

Sol independently verified the final 54 cues: normalized ASR text is unchanged, every boundary equals the original source offset plus `ceil(word_seconds * 1000)`, and differing supported speaker labels are not merged. Evidence: `var/ep11-pilot-validation-20260906/final-real-evidence-integrity.json`, checksum `4fd35b32267a5dde8624dc9d91f969729c56fcd58f7a45138f103f36e3eb77b5`. The final assessment binds both builder and pilot code hashes. Earlier draft directories remain intact.

## GPU execution and retained failures

All attempt directories below are under `var/`; durations derive from first-to-last event monotonic timestamps and are not GPU invoices.

| Attempt suffix | Event span, seconds | Outcome |
|---|---:|---|
| 20260906T070242Z-871eefb1 | 145.641 | uv unavailable; production venv had no pip; temporary Pod externally ABSENT |
| 20260906T070715Z-a4401c08 | 178.125 | uv absent from persistent PATH too; temporary Pod externally ABSENT |
| 20260906T071229Z-c47e63b1 | 6.813 | Null Ada offer; no Pod created; null-price handling repaired |
| 20260906T071440Z-e228bd45 | 140.141 | Rootless tar could not restore archive ownership; repaired with --no-same-owner; Pod externally ABSENT |
| 20260906T071815Z-2ebd6c08 | 185.859 | Isolated install succeeded; missing numba stopped import; Pod externally ABSENT |
| 20260906T072407Z-7032e0c9 | 282.625 | All five clips completed on NVIDIA L4; Pod ptk1esg35zdh8r externally ABSENT |

Each directory starts with `disposable-asr-pilot-`. The successful transfer/worker took 272.36 seconds, including readiness, uploads, isolated installation, model work and verified result retrieval. Pure model inference duration was not separately instrumented and is UNKNOWN. Raw archive: 21915 bytes, SHA-256 `2e2a0a3a697393ba2b7446aba00e15457deab1e7cd88998b6b67d994e4a18459`.

The pinned uv 0.8.14 Linux installer was downloaded locally and checked against the official release digest before GPU use. Installer receipt: `var/pilot-installer-uv-0.8.14/receipt.json`. Installed packages stayed in the isolated pilot directory. Production venv and source media were preserved. The successful isolated runtime inventory includes stable-ts 2.19.1, openai-whisper 20250625, numba 0.67.0, llvmlite 0.49.0 and NumPy 2.4.6, alongside retained CUDA PyTorch and faster-whisper.

The L4 price was USD 0.49/hour. The controller retained its 550-second lease, 120-second shutdown reserve and USD 1.10 minimum remaining balance. The successful lease's calculated maximum, including observed USD 0.005/hour account spend, was USD 0.075625. This is a bound, not a charged-cost claim.

Fresh external provider snapshot: displayed balance USD 3.6213915897; only retained Pod 781ct55zv4gkle, EXITED; volume xgogcmey5o, 50 GB, preserved. Evidence: `var/ep11-pilot-validation-20260906/provider-snapshot.json`. No top-up, GPT transcription API, Drive mutation, volume deletion, stable tag or Episode 12 run occurred.

## Notification blocker and continuation

Gmail accepted the first attempt's eight stage submissions. Later submissions timed out. A subsequent bounded root attempt returned SMTP `550 5.4.5 Daily user sending limit exceeded`. Evidence: `var/ep11-pilot-takeover-20260906-events.json`. Later stage events were retained locally with an explicit blocked-mail reason, without repeated quota-exhausted SMTP calls. Do not report them as sent or inbox-confirmed.

Next action is acoustic/text review of this actual five-clip candidate, especially the flagged cues. The available machine diagnostics do not authorize pilot acceptance. Do not repeat GPU inference merely to replay the SRT, and do not start Episode 12 before acceptance. Local deterministic replay, into a new unused output directory:

```powershell
& .\.venv\Scripts\python.exe -m var.build_contextual_asr_pilot var/disposable-asr-pilot-20260906T072407Z-7032e0c9/asr/raw NEW_OUTPUT_DIRECTORY
```
