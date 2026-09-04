from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml

from v010_core import FILES as CORE_FILES
from v010_stages import FILES as STAGE_FILES

STATIC_FILES: dict[str, str] = {
"pyproject.toml": '''[build-system]
requires = ["setuptools==75.8.0", "wheel==0.45.1"]
build-backend = "setuptools.build_meta"

[project]
name = "ma-sub"
version = "0.1.0"
description = "Resumable Muhtemel Ask Turkish-to-Indonesian subtitle pipeline"
requires-python = ">=3.10,<3.13"
dependencies = [
  "PyYAML==6.0.2",
  "openpyxl==3.1.5",
  "yt-dlp==2025.1.26",
]

[project.optional-dependencies]
test = ["pytest==8.3.5"]
gpu = ["faster-whisper==1.1.1", "whisperx==3.3.1"]

[project.scripts]
mas = "mas.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["mas*"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
pythonpath = ["src"]
markers = [
  "gpu: requires an NVIDIA GPU and model downloads",
  "integration: multi-stage pipeline integration test",
]
''',
"requirements.lock": '''PyYAML==6.0.2
openpyxl==3.1.5
pytest==8.3.5
setuptools==75.8.0
wheel==0.45.1
yt-dlp==2025.1.26
''',
"requirements-gpu.lock": '''ctranslate2==4.4.0
faster-whisper==1.1.1
whisperx==3.3.1
''',
"DEPENDENCIES.md": '''# Runtime dependency contract

- Python: 3.11 in CI and the RunPod bootstrap virtual environment.
- CUDA image: CUDA 12.4 / cuDNN 9 runtime.
- PyTorch image baseline: 2.5.1 CUDA 12.4.
- ffmpeg and ffprobe: distribution-pinned packages from the immutable container base layer.
- faster-whisper: 1.1.1.
- WhisperX: 3.3.1, installed for compatibility with the preserved V2 alignment work. The v0.1.0 production alignment adapter uses retained faster-whisper word timing and does not silently invoke a changing WhisperX API.
- yt-dlp: 2025.1.26.

All Python direct dependencies are exact pins. A real GPU build and full-episode inference remain release-test items outside CPU CI and must be recorded before a later stable release claims GPU validation.
''',
"Dockerfile": '''FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG INSTALL_GPU=1
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MAS_HOME=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ma-sub
COPY requirements.lock requirements-gpu.lock ./
RUN python -m pip install --no-cache-dir -r requirements.lock \
    && if [ "$INSTALL_GPU" = "1" ]; then python -m pip install --no-cache-dir -r requirements-gpu.lock; fi

COPY . .
RUN chmod +x mas runpod/bootstrap.sh runpod/start.sh runpod/preflight.sh \
    && python -m compileall -q src

ENTRYPOINT ["./mas"]
CMD ["--help"]
''',
".dockerignore": '''.git
.venv
__pycache__
.pytest_cache
*.pyc
EPISODES
bootstrap
release-tools
''',
".gitignore": '''.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
EPISODES/
*.partial
.env
''',
"runpod/bootstrap.sh": '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade "pip==25.0.1"
.venv/bin/python -m pip install -r requirements.lock
if [[ "${MAS_INSTALL_GPU:-1}" == "1" ]]; then
  .venv/bin/python -m pip install -r requirements-gpu.lock
fi
chmod +x mas runpod/*.sh
./mas doctor
''',
"runpod/preflight.sh": '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
./mas doctor --strict-gpu
''',
"runpod/start.sh": '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  ./runpod/bootstrap.sh
fi
exec ./mas "$@"
''',
"runpod/README.md": '''# RunPod

Use an NVIDIA template with at least 16 GB GPU VRAM and a persistent volume mounted at `/workspace`.

```bash
cd /workspace
git clone https://github.com/ismailkorkmaz1905/ma-sub.git
cd ma-sub
git checkout v0.1.0
./runpod/bootstrap.sh
./mas doctor --strict-gpu
./mas run 13 --source-url "SOURCE_URL"
```

Set `MAS_YTDLP_COOKIES=/workspace/cookies.txt` only when the source host requires cookies. Episode state and expensive ASR chunk outputs live under `EPISODES/.../work`, so keep the repository or set `MAS_HOME` to a persistent mount.

Exit code 20 means the Turkish correction ZIP is required. Exit code 21 means the Indonesian translation ZIP is required. Both are normal handoff states, not lost work.
''',
".github/workflows/test.yml": '''name: test

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:

permissions:
  contents: read

jobs:
  cpu-regression:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11.11"
          cache: pip
      - name: System dependencies
        run: sudo apt-get update && sudo apt-get install -y ffmpeg
      - name: Install locked dependencies
        run: python -m pip install -r requirements.lock
      - name: Compile
        run: python -m compileall -q src tests
      - name: Test
        run: ./mas test -- -q
      - name: CLI smoke
        run: |
          ./mas --help
          ./mas doctor
          ./mas clean 999 --dry-run
      - name: Docker CPU smoke build
        run: docker build --build-arg INSTALL_GPU=0 -t ma-sub:${{ github.sha }} .
''',
"README.md": '''# ma-sub

## Episode 13 is out. What do I do?

1. Start a RunPod NVIDIA GPU pod with `/workspace` on persistent storage.
2. Clone the stable release once, or update an existing checkout:

```bash
cd /workspace
git clone https://github.com/ismailkorkmaz1905/ma-sub.git
cd ma-sub
git checkout v0.1.0
./runpod/bootstrap.sh
```

3. Start or resume the episode:

```bash
./mas run 13 --source-url "SOURCE_URL"
```

The pipeline acquires the source, extracts audio, preserves Turkish captions as separate evidence, detects speech, runs chunked Turkish ASR, reconciles evidence, creates targeted audio-review clips, and writes:

```text
EPISODES/Muhtemel Ask 13.Bolum/translation_input/
  Muhtemel Ask 13.Bolum_TR_CORRECTION_PACK.zip
  TR_PROMPT.txt
```

4. When the console prints `GPU WORK COMPLETE`, stop the GPU pod. Give the ZIP and generated prompt to ChatGPT Pro.
5. Put ChatGPT's exact returned file here:

```text
EPISODES/Muhtemel Ask 13.Bolum/translation_output/
  Muhtemel Ask 13.Bolum_TR_TEXT_CORRECTED.zip
```

6. Run the same command again:

```bash
./mas run 13
```

It validates every immutable field, aligns corrected Turkish using retained speech/word timing evidence, then creates the Indonesian translation ZIP and `ID_PROMPT.txt`. Give those to ChatGPT Pro and place the exact return file at:

```text
EPISODES/Muhtemel Ask 13.Bolum/translation_output/
  Muhtemel Ask 13.Bolum_ID_TRANSLATED.zip
```

7. Run the same command again:

```bash
./mas run 13
```

Final files appear under:

```text
EPISODES/Muhtemel Ask 13.Bolum/final/subtitles/
  Muhtemel Ask 13.Bolum-tr.srt
  Muhtemel Ask 13.Bolum-id.srt
```

QC is written to `reports/qc.json` and `reports/qc.md`. The Indonesian SRT is mandatory. MKV muxing uses stream copy and a mux failure does not delete valid SRT files.

## Normal commands

```bash
./mas run 13 --source-url "SOURCE_URL"  # start
./mas run 13                             # resume
./mas status 13                          # checkpoints and wait state
./mas status 13 --json                   # machine-readable state
./mas doctor --strict-gpu                # RunPod readiness
./mas test                               # full deterministic suite
./mas clean 13 --dry-run                 # preview regenerable caches
./mas clean 13 --execute                 # remove only listed caches
```

`clean` never removes source media, translation inputs/outputs, reports, or final files.

## Recovery after an error

Read the concise `CAUSE` line, then inspect the traceback path printed under `logs/`. Fix the stated dependency/input problem and rerun:

```bash
./mas run 13
```

A stage is skipped only when its input fingerprint, scoped configuration, implementation fingerprint, output size, and SHA-256 all still match. A changed Indonesian return ZIP does not rerun ASR. A changed source invalidates its downstream stages.

## Source options

- HTTP/YouTube-like URL: pass `--source-url`. `yt-dlp` also saves available Turkish subtitles as evidence.
- Protected URL: set `MAS_YTDLP_COOKIES=/workspace/cookies.txt` once.
- Local file: pass a path or `file:///...` URL.
- Existing episode source: place one supported video under the episode `source/` directory, then run without `--source-url`.

## Updating and rolling back

Do not run weekly episodes from an untagged moving branch. Update to the newest tested tag explicitly:

```bash
git fetch --tags origin
git checkout v0.1.0
./mas doctor
```

Roll back without touching episode data:

```bash
git checkout v0.1.0
./mas run 13
```

Every state file records pipeline version, schema version, rules version, Git commit, stage fingerprints, output hashes, and runtimes.

## What v0.1.0 means

This is the first production-oriented CLI baseline. CPU/offline checkpoint, ZIP-invariant, SRT, QC, and Docker smoke paths are tested. It is not a claim that a real full episode has completed GPU inference. Do not call this `v1.0.0` until that happens.

The preserved V2 notebooks and source references are under `legacy/V2_BETA/`. They are reference material and are not production entry points.
''',
"CHANGELOG.md": '''# Changelog

## 0.1.0 - 2026-09-05

- Migrated the validated V2 deterministic logic and tests into a normal Python package.
- Added one-command CLI operation, hash-verified checkpoints, scoped downstream invalidation, structured logs, and safe cleanup.
- Added real source/media/captions/VAD/chunked faster-whisper stage implementations.
- Added strict ChatGPT Pro Turkish-correction and Indonesian-translation ZIP handoffs without OpenAI API billing.
- Added Turkish and Indonesian SRT rendering, optional stream-copy MKV muxing, machine/human QC, and archive validation.
- Added pinned RunPod/Docker setup, CI, V2 rule inventory, canonical names/glossary extraction, and offline interruption/resume regression coverage.
''',
"MIGRATION_NOTES.md": '''# V2 to CLI migration notes

The migration treats the Drive V2 package as source of truth. The release builder inventories every rule keyword requested by the operator, copies all located V2 documentation/notebooks into `legacy/V2_BETA/reference`, archives every production file replaced during extraction, and leaves non-notebook deterministic V2 modules/tests in place.

Production entry points are `./mas` and `src/mas/cli.py`. Notebooks are not imported by production code.

Rule conflicts are recorded in `rules/V2_RULE_INVENTORY.md`. Explicit project/translation instructions have precedence. Where numeric limits conflict without explicit precedence, the stricter safe limit is selected and its sources are recorded in `config/pipeline.yaml`.

The v0.1.0 alignment adapter uses word-level speech timing already produced by faster-whisper. Preserved V2/WhisperX work remains available for a later proven adapter. This avoids claiming an untested changing API path while still preventing fixed blind padding.

A real full-episode GPU run has not been represented as completed by this migration. GPU inference, model download, protected-source cookies, and full-duration memory/runtime behavior must be validated on the first RunPod episode test.
''',
"rules/SUBTITLE_SPEC.md": '''# Subtitle specification

Rules version: 0.1.0

The generated `config/pipeline.yaml` contains numeric limits extracted from V2 sources. When V2 sources conflict without explicit precedence, the migration chooses the stricter safe threshold and records all candidates.

Mandatory invariants:

- Timestamps increase and every end is greater than its start.
- Illegal overlap is reported. Separate speakers are never merged merely to improve line length.
- Speech/word evidence controls boundaries. Do not apply blind fixed padding to every block.
- Keep at most the configured line count and configured characters per line. Report CPS, too-short, and too-long blocks.
- Dialogue cannot be blank or duplicated.
- Final SRT contains no speaker labels, audit data, hashes, UIDs, schema fields, or translator notes.
- Readability formatting may change line breaks but cannot alter meaning.
- Recoverable uncertainty is preserved in `review_required.json`; mandatory structural failures are fatal.
- Overlapping dialogue follows preserved V2 evidence and is never aggressively collapsed when diarization is uncertain.
''',
"rules/TRANSLATION_SPEC.md": '''# Turkish correction and Indonesian translation specification

Rules version: 0.1.0

## Turkish correction

Correct ASR text only where text evidence supports it. Preserve Turkish diacritics, punctuation, meaning, register, hesitation, interruptions, unfinished speech, names, numbers, dates, quantities, and currencies. Never claim to have listened to audio. Records flagged as audio-dependent remain unchanged and flagged unless `audio_reviewed` is already true.

## Indonesian translation

Translate as natural spoken Indonesian, not generic literal machine translation. Use `aku`, `kamu`, `nggak`, `udah`, and `aja` when the relationship and scene permit. Use `saya`, `Anda`, `Pak`, and `Bu` in formal contexts. Do not force slang into formal scenes.

Preserve relationship dynamics, intimacy, anger, sarcasm, humor, insults, hesitation, interruptions, unfinished sentences, and emotional tone. Do not sanitize dialogue, explain jokes, add translator notes, add speaker labels, move dialogue between blocks, split/merge/reorder records, or alter proper names, numbers, dates, quantities, and currencies.

`rules/NAMES.json` and `rules/GLOSSARY.json` are generated from located authoritative V2 source mappings and are canonical for the versioned release.
''',
"rules/GLOSSARY.json": '''{"schema_version":"1.0","rules_version":"0.1.0","entries":[],"generated_from_v2":true}
''',
"rules/NAMES.json": '''{"schema_version":"1.0","rules_version":"0.1.0","entries":[],"generated_from_v2":true}
''',
"tests/production/conftest.py": '''from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Callable

import pytest


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MAS_HOME", str(tmp_path))
    monkeypatch.setenv("MAS_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
    monkeypatch.delenv("MAS_CONFIG", raising=False)
    monkeypatch.delenv("MAS_FAIL_AFTER_STAGE", raising=False)
    return tmp_path


def rewrite_pack(
    source: Path,
    target: Path,
    *,
    mutate_manifest: Callable[[dict], None] | None = None,
    mutate_records: Callable[[list[dict]], None] | None = None,
) -> None:
    with zipfile.ZipFile(source) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    manifest = json.loads(members["manifest.json"])
    records = json.loads(members["records.json"])
    if mutate_manifest:
        mutate_manifest(manifest)
    if mutate_records:
        mutate_records(records)
    members["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    members["records.json"] = json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def make_tr_return(pack: Path, target: Path) -> None:
    def mutate(records: list[dict]) -> None:
        for row in records:
            row["tr_text_corrected"] = row.get("tr_text", "")
    rewrite_pack(pack, target, mutate_records=mutate)


def make_id_return(pack: Path, target: Path, suffix: str = "") -> None:
    translations = {
        "Merhaba Defne.": "Halo Defne.",
        "Nasılsın?": "Apa kabar?",
        "250 lira borcum kaldı.": "Utangku tinggal 250 lira.",
    }

    def mutate(records: list[dict]) -> None:
        for row in records:
            source = row.get("tr_text", "")
            row["id_text"] = translations.get(source, source) + (suffix if "250" not in source else "")
    rewrite_pack(pack, target, mutate_records=mutate)
''',
"tests/production/test_pack_invariants.py": '''from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from mas.errors import InvariantError
from mas.packs.build import build_pack
from mas.packs.validate import validate_returned
from tests.production.conftest import rewrite_pack


def records() -> list[dict]:
    return [
        {
            "block_uid": "uid-1",
            "block_index": 1,
            "start_ms": 1000,
            "end_ms": 2500,
            "tr_text": "Defne'ye 250 TL verdim.",
            "id_text": "",
            "asr_audit": {"confidence": 0.9},
            "review_required": False,
            "audio_reviewed": False,
            "review_disposition": None,
        },
        {
            "block_uid": "uid-2",
            "block_index": 2,
            "start_ms": 2700,
            "end_ms": 4200,
            "tr_text": "Görüşürüz.",
            "id_text": "",
            "asr_audit": {"confidence": 0.8},
            "review_required": False,
            "audio_reviewed": False,
            "review_disposition": None,
        },
    ]


def prepared(tmp_path: Path, kind: str = "ID") -> tuple[Path, dict, list[dict]]:
    source = records()
    pack = tmp_path / "pack.zip"
    manifest = build_pack(pack, 13, kind, source)
    return pack, manifest, source


def valid_return(tmp_path: Path, kind: str = "ID") -> tuple[Path, dict, list[dict]]:
    pack, manifest, source = prepared(tmp_path, kind)
    target = tmp_path / "returned.zip"

    def mutate(rows: list[dict]) -> None:
        for row in rows:
            if kind == "TR":
                row["tr_text_corrected"] = row["tr_text"] + "."
            else:
                row["id_text"] = "Saya memberi Defne 250 TL." if row["block_uid"] == "uid-1" else "Sampai jumpa."
    rewrite_pack(pack, target, mutate_records=mutate)
    return target, manifest, source


def test_valid_id_return(tmp_path: Path) -> None:
    target, manifest, source = valid_return(tmp_path)
    result = validate_returned(target, manifest, source)
    assert [row["block_uid"] for row in result] == ["uid-1", "uid-2"]
    assert result[0]["id_text"] == "Saya memberi Defne 250 TL."


def test_punctuation_only_tr_correction(tmp_path: Path) -> None:
    target, manifest, source = valid_return(tmp_path, "TR")
    result = validate_returned(target, manifest, source)
    assert result[0]["tr_text"].endswith(".")
    assert "Görüşürüz" in result[1]["tr_text"]


@pytest.mark.parametrize(
    "manifest_change,record_change,error",
    [
        (lambda value: value.__setitem__("episode", 12), None, "episode"),
        (lambda value: value.__setitem__("schema_sha256", "0" * 64), None, "schema_sha256"),
        (lambda value: value.__setitem__("input_sha256", "1" * 64), None, "input_sha256"),
        (None, lambda rows: rows[0].__setitem__("block_uid", "changed"), "block_uid"),
        (None, lambda rows: rows[0].__setitem__("block_index", 9), "block_index"),
        (None, lambda rows: rows[0].__setitem__("start_ms", 999), "immutable"),
        (None, lambda rows: rows[0]["asr_audit"].__setitem__("confidence", 0.1), "immutable"),
        (None, lambda rows: rows.pop(), "record count"),
        (None, lambda rows: rows.append(dict(rows[-1])), "record count"),
        (None, lambda rows: rows.reverse(), "block_uid"),
    ],
)
def test_rejects_invariant_changes(tmp_path: Path, manifest_change, record_change, error: str) -> None:
    pack, manifest, source = prepared(tmp_path)
    target = tmp_path / "bad.zip"

    def translated(rows: list[dict]) -> None:
        for row in rows:
            row["id_text"] = "Saya memberi Defne 250 TL." if row["block_uid"] == "uid-1" else "Sampai jumpa."
        if record_change:
            record_change(rows)
    rewrite_pack(pack, target, mutate_manifest=manifest_change, mutate_records=translated)
    with pytest.raises(InvariantError, match=error):
        validate_returned(target, manifest, source)


def test_rejects_number_or_currency_change(tmp_path: Path) -> None:
    pack, manifest, source = prepared(tmp_path)
    target = tmp_path / "bad-number.zip"

    def mutate(rows: list[dict]) -> None:
        rows[0]["id_text"] = "Saya memberi Defne 300 USD."
        rows[1]["id_text"] = "Sampai jumpa."
    rewrite_pack(pack, target, mutate_records=mutate)
    with pytest.raises(InvariantError, match="numbers changed"):
        validate_returned(target, manifest, source)


def test_rejects_non_zip(tmp_path: Path) -> None:
    _, manifest, source = prepared(tmp_path)
    target = tmp_path / "wrong.zip"
    target.write_text("not a zip", encoding="utf-8")
    with pytest.raises(InvariantError, match="valid ZIP"):
        validate_returned(target, manifest, source)


def test_rejects_zip_traversal(tmp_path: Path) -> None:
    _, manifest, source = prepared(tmp_path)
    target = tmp_path / "traversal.zip"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("../records.json", "[]")
        archive.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(InvariantError, match="unsafe ZIP"):
        validate_returned(target, manifest, source)
''',
"tests/production/test_subtitle_quality.py": '''from __future__ import annotations

import json
from pathlib import Path

import pytest

from mas.config import Settings
from mas.context import EpisodeContext
from mas.stages import reconcile
from mas.subtitle.segmentation import wrap_text
from mas.subtitle.speakers import may_merge, strip_visible_speaker_label
from mas.subtitle.srt import render
from mas.subtitle.validation import qc


def row(uid: str, index: int, start: int, end: int, text: str, speaker: str = "A") -> dict:
    return {
        "block_uid": uid,
        "block_index": index,
        "start_ms": start,
        "end_ms": end,
        "tr_text": text,
        "id_text": text,
        "speaker": speaker,
        "flags": [],
    }


def test_srt_renderer_and_speaker_label_removal() -> None:
    text = render([row("a", 1, 1000, 2500, "DEFNE: Merhaba")], "tr_text")
    assert "00:00:01,000 --> 00:00:02,500" in text
    assert "DEFNE:" not in text
    assert "Merhaba" in text


def test_no_metadata_leak() -> None:
    with pytest.raises(Exception, match="metadata leaked"):
        render([row("a", 1, 1000, 2500, "schema_sha256: abc")], "tr_text")


def test_overlap_rapid_switch_and_separate_speakers() -> None:
    left = row("a", 1, 1000, 2200, "Merhaba", "A")
    right = row("b", 2, 2100, 3000, "Nasılsın?", "B")
    issues = qc([left, right], Settings(), "id_text")
    assert any(issue["kind"] == "overlap" for issue in issues)
    assert not may_merge(left, right)


def test_line_length_and_cps() -> None:
    long = row("a", 1, 1000, 1800, "Bu çok uzun ve okunması zor olan bir altyazı satırıdır")
    issues = qc([long], Settings(max_cps=10, max_line_chars=20), "id_text")
    assert {issue["kind"] for issue in issues} >= {"cps", "line_length"}
    assert "\n" in wrap_text(long["id_text"], 20, 2)


def test_turkish_diacritics_and_indonesian_register_are_in_rules() -> None:
    root = Path(__file__).resolve().parents[2]
    translation = (root / "rules" / "TRANSLATION_SPEC.md").read_text(encoding="utf-8")
    assert "diacritics" in translation
    assert all(token in translation for token in ("aku", "kamu", "nggak", "udah", "aja", "saya", "Anda", "Pak", "Bu"))


def _write_reconcile_inputs(ctx: EpisodeContext, asr_rows: list[dict], captions: list[dict], speech: list[dict]) -> None:
    ctx.ensure_dirs()
    (ctx.root / "work" / "asr").mkdir(parents=True, exist_ok=True)
    (ctx.root / "work" / "asr" / "asr.json").write_text(json.dumps({"records": asr_rows}), encoding="utf-8")
    (ctx.root / "work" / "captions.json").write_text(json.dumps({"captions": captions}), encoding="utf-8")
    (ctx.root / "work" / "vad.json").write_text(json.dumps({"speech": speech}), encoding="utf-8")


def test_empty_caption_source_is_recoverable(workspace: Path) -> None:
    ctx = EpisodeContext(20, Settings(), fixture=True)
    _write_reconcile_inputs(ctx, [row("a", 1, 1000, 2500, "Merhaba")], [], [{"start_ms": 1000, "end_ms": 2500}])
    result = reconcile.run(ctx, {})
    payload = json.loads((ctx.root / "work" / "records.json").read_text())
    assert payload["record_count"] == 1
    assert result["metadata"]["review_count"] == 0


def test_caption_only_and_vad_only_are_recorded_for_review(workspace: Path) -> None:
    ctx = EpisodeContext(21, Settings(unresolved_speech_min_seconds=0.5), fixture=True)
    captions = [
        {
            "caption_uid": "caption-1",
            "start_ms": 1000,
            "end_ms": 2200,
            "text": "Defne geldi.",
            "source_path": "source/test.vtt",
        }
    ]
    speech = [{"start_ms": 1000, "end_ms": 2200}, {"start_ms": 4000, "end_ms": 5000}]
    _write_reconcile_inputs(ctx, [], captions, speech)
    reconcile.run(ctx, {})
    records = json.loads((ctx.root / "work" / "records.json").read_text())["records"]
    review = json.loads((ctx.root / "work" / "review_required.json").read_text())["items"]
    assert records[0]["flags"] == ["orphan_youtube_caption"]
    reasons = {reason for item in review for reason in item["reasons"]}
    assert {"orphan_youtube_caption", "unresolved_vad_speech"} <= reasons


def test_speaker_label_helper() -> None:
    assert strip_visible_speaker_label("DEFNE: Merhaba") == "Merhaba"
''',
"tests/production/test_pipeline_resume.py": '''from __future__ import annotations

import json
from pathlib import Path

import pytest

from mas.errors import InjectedFailure, InvariantError
from mas.pipeline import run
from tests.production.conftest import make_id_return, make_tr_return


def episode_root(workspace: Path, episode: int) -> Path:
    return workspace / "EPISODES" / f"Muhtemel Ask {episode}.Bolum"


@pytest.mark.integration
def test_interruption_handoffs_resume_and_targeted_invalidation(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode = 71
    root = episode_root(workspace, episode)
    monkeypatch.setenv("MAS_FAIL_AFTER_STAGE", "asr")
    with pytest.raises(InjectedFailure, match="injected interruption"):
        run(episode, fixture=True)
    interrupted = json.loads((root / "work" / "state.json").read_text())
    assert interrupted["stages"]["asr"]["status"] == "complete"
    asr_fingerprint = interrupted["stages"]["asr"]["input_fingerprint"]
    monkeypatch.delenv("MAS_FAIL_AFTER_STAGE")

    assert run(episode, fixture=True) == 20
    waiting_tr = json.loads((root / "work" / "state.json").read_text())
    assert waiting_tr["status"] == "WAITING_TR"
    assert waiting_tr["stages"]["asr"]["input_fingerprint"] == asr_fingerprint

    make_tr_return(
        root / "translation_input" / f"Muhtemel Ask {episode}.Bolum_TR_CORRECTION_PACK.zip",
        root / "translation_output" / f"Muhtemel Ask {episode}.Bolum_TR_TEXT_CORRECTED.zip",
    )
    assert run(episode, fixture=True) == 21
    waiting_id = json.loads((root / "work" / "state.json").read_text())
    align_fingerprint = waiting_id["stages"]["align"]["input_fingerprint"]

    id_pack = root / "translation_input" / f"Muhtemel Ask {episode}.Bolum_ID_TRANSLATION_PACK.zip"
    id_return = root / "translation_output" / f"Muhtemel Ask {episode}.Bolum_ID_TRANSLATED.zip"
    make_id_return(id_pack, id_return)
    assert run(episode, fixture=True) == 0
    complete = json.loads((root / "work" / "state.json").read_text())
    assert complete["status"] == "PASS"
    assert (root / "final" / "subtitles" / f"Muhtemel Ask {episode}.Bolum-id.srt").exists()
    first_id_hash = complete["stages"]["id_return"]["metadata"]["return_zip_sha256"]

    assert run(episode, fixture=True) == 0
    stable = json.loads((root / "work" / "state.json").read_text())
    assert stable["stages"]["asr"]["input_fingerprint"] == asr_fingerprint
    assert stable["stages"]["align"]["input_fingerprint"] == align_fingerprint

    make_id_return(id_pack, id_return, suffix=" sekarang")
    assert run(episode, fixture=True) == 0
    changed = json.loads((root / "work" / "state.json").read_text())
    assert changed["stages"]["id_return"]["metadata"]["return_zip_sha256"] != first_id_hash
    assert changed["stages"]["align"]["input_fingerprint"] == align_fingerprint
    assert changed["stages"]["asr"]["input_fingerprint"] == asr_fingerprint


@pytest.mark.integration
def test_stale_checkpoint_and_source_change_invalidate_downstream(workspace: Path) -> None:
    episode = 72
    root = episode_root(workspace, episode)
    assert run(episode, fixture=True) == 20
    state1 = json.loads((root / "work" / "state.json").read_text())
    first_source_output = root / state1["stages"]["source"]["outputs"][0]["path"]
    first_source_output.write_text(first_source_output.read_text() + "\n", encoding="utf-8")
    assert run(episode, fixture=True) == 20
    state2 = json.loads((root / "work" / "state.json").read_text())
    assert state2["stages"]["source"]["outputs"][0]["sha256"] != state1["stages"]["source"]["outputs"][0]["sha256"]
    assert state2["stages"]["asr"]["input_fingerprint"] != state1["stages"]["asr"]["input_fingerprint"]


@pytest.mark.integration
def test_wrong_old_return_is_rejected_after_source_change(workspace: Path) -> None:
    episode = 73
    root = episode_root(workspace, episode)
    assert run(episode, fixture=True) == 20
    tr_pack = root / "translation_input" / f"Muhtemel Ask {episode}.Bolum_TR_CORRECTION_PACK.zip"
    tr_return = root / "translation_output" / f"Muhtemel Ask {episode}.Bolum_TR_TEXT_CORRECTED.zip"
    make_tr_return(tr_pack, tr_return)
    records = root / "source" / "fixture_records.json"
    payload = json.loads(records.read_text())
    payload[0]["tr_text"] = "Değişti."
    records.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(InvariantError, match="input_sha256"):
        run(episode, fixture=True)
''',
"tests/production/test_names_and_rules.py": '''from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NAMES = json.loads((ROOT / "rules" / "NAMES.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "entry",
    NAMES.get("entries", []),
    ids=lambda entry: str(entry.get("canonical") or entry.get("source_form")),
)
def test_every_extracted_canonical_name(entry: dict) -> None:
    assert isinstance(entry.get("canonical"), str) and entry["canonical"].strip()
    assert entry.get("variants") or entry.get("source_form")
    assert entry.get("sources")


def test_rule_inventory_and_required_v2_references() -> None:
    inventory = (ROOT / "rules" / "V2_RULE_INVENTORY.md").read_text(encoding="utf-8")
    assert "block_uid" in inventory
    assert "schema_sha256" in inventory
    assert "speaker" in inventory.lower()
    manifest = json.loads((ROOT / "legacy" / "V2_BETA" / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert not manifest["missing_required"]
    assert len(manifest["files"]) >= 10
''',
}

REQUIRED_V2 = [
    "00_README_FIRST.md",
    "V2_BETA_README.md",
    "V2_PROJECT_INSTRUCTIONS.md",
    "V2_NEW_EPISODE_PROMPTS.md",
    "TRANSLATION_INSTRUCTIONS.md",
    "V2_TEST_REPORT.md",
    "TEST_REPORT.md",
    "01_PREPARE_TR.ipynb",
    "02_ALIGN_PREPARE_ID.ipynb",
    "03_FINALIZE_V2.ipynb",
    "04_ARCHIVE_V2.ipynb",
    "requirements-v2-colab.txt",
]

RULE_KEYWORDS = [
    "proper name",
    "character name",
    "canonical",
    "turkish correction",
    "indonesian translation",
    "religious",
    "currency",
    "duration",
    "line length",
    "timing",
    "speaker",
    "vad",
    "hallucination",
    "youtube caption",
    "immutable",
    "hash validation",
    "schema validation",
    "review_required",
    "alignment provenance",
    "audio review",
    "block_uid",
    "block_index",
    "schema_sha256",
    "schema_version",
    "unresolved_vad_speech",
    "suspected_asr_hallucination",
    "orphan_youtube_caption",
    "audio_reviewed",
    "review_disposition",
]

TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ipynb",
    ".csv",
    ".tsv",
}


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    result: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe bootstrap archive member: {member.name}")
        result.append(member)
    return result


def _merge_bootstrap_archive(root: Path) -> list[str]:
    parts = sorted((root / "bootstrap").glob("essential.part-*"))
    if not parts:
        return []
    encoded = "".join(path.read_text(encoding="utf-8").strip() for path in parts)
    try:
        data = base64.b64decode(encoded, validate=True)
    except Exception:
        return []
    merged: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ma-sub-bootstrap-") as temp_name:
        temp = Path(temp_name)
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as archive:
                archive.extractall(temp, members=_safe_members(archive), filter="data")
        except (tarfile.TarError, TypeError):
            return []
        for source in temp.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(temp)
            if relative.parts and relative.parts[0] in {".git", "bootstrap", "release-tools", "EPISODES"}:
                continue
            target = root / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                merged.append(str(relative))
    return merged


def _iter_source_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in {".git", "release-tools", "EPISODES", ".venv"}:
            continue
        if "__pycache__" in relative.parts:
            continue
        yield path


def _find_required(root: Path, name: str) -> Path | None:
    candidates = [path for path in root.rglob(name) if path.is_file() and "release-tools" not in path.parts]
    if not candidates:
        return None
    candidates.sort(key=lambda path: ("legacy" in path.parts, len(path.parts), str(path)))
    return candidates[0]


def _archive_replaced(root: Path, paths: Iterable[str]) -> None:
    base = root / "legacy" / "V2_BETA" / "overwritten"
    for relative in sorted(set(paths)):
        source = root / relative
        if not source.exists() or not source.is_file():
            continue
        target = base / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_references(root: Path) -> dict[str, Any]:
    reference_root = root / "legacy" / "V2_BETA" / "reference"
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    from hashlib import sha256

    for name in REQUIRED_V2:
        source = _find_required(root, name)
        if source is None:
            missing.append(name)
            continue
        relative = source.relative_to(root)
        target = reference_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        digest = sha256(target.read_bytes()).hexdigest()
        files.append({"required_name": name, "source_path": str(relative), "preserved_path": str(target.relative_to(root)), "sha256": digest, "size": target.stat().st_size})
    for notebook in root.rglob("*.ipynb"):
        if "release-tools" in notebook.parts or reference_root in notebook.parents:
            continue
        target = reference_root / "notebooks" / notebook.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(notebook, target)
    manifest = {"source": "Muhtemel_Ask_Subtitle_System_V2_BETA_2026-09-04.zip", "files": files, "missing_required": missing}
    manifest_path = root / "legacy" / "V2_BETA" / "SOURCE_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _rule_inventory(root: Path) -> tuple[str, list[Path]]:
    entries: list[tuple[str, int, str, list[str]]] = []
    scanned: list[Path] = []
    lowered_keywords = [(keyword, keyword.lower()) for keyword in RULE_KEYWORDS]
    for path in sorted(_iter_source_files(root), key=lambda value: str(value)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned.append(path)
        for number, line in enumerate(text.splitlines(), 1):
            compact = line.strip()
            if not compact:
                continue
            lowered = compact.lower()
            matches = [original for original, keyword in lowered_keywords if keyword in lowered]
            if matches:
                entries.append((str(path.relative_to(root)), number, compact[:500], matches))
    lines = [
        "# V2 rule inventory",
        "",
        "Generated from the complete located V2 project tree before production files were replaced.",
        "",
        f"Scanned text files: {len(scanned)}",
        f"Matched rule lines: {len(entries)}",
        "",
        "Precedence: explicit V2 project/translation instructions override implementation comments and tests. If numeric safety limits conflict without explicit precedence, the stricter safe limit is selected and every candidate remains listed here/config provenance.",
        "",
    ]
    for path, number, text, keywords in entries:
        lines.append(f"- `{path}:{number}` [{', '.join(keywords)}] {text}")
    return "\n".join(lines) + "\n", scanned


def _flatten(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(child, name)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _structured_candidates(paths: Iterable[Path]) -> list[tuple[str, Any, str]]:
    result: list[tuple[str, Any, str]] = []
    for path in paths:
        if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            else:
                payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for key, value in _flatten(payload):
            result.append((key.lower(), value, str(path)))
    return result


def _select_thresholds(root: Path, source_files: list[Path]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    defaults: dict[str, Any] = {
        "min_duration_ms": 700,
        "max_duration_ms": 7000,
        "max_cps": 20.0,
        "max_line_chars": 42,
        "max_lines": 2,
        "max_overlap_ms": 0,
        "asr_chunk_seconds": 900,
        "vad_min_silence_seconds": 0.35,
        "review_clip_padding_ms": 750,
    }
    aliases = {
        "min_duration_ms": ("min_duration_ms", "minimum_duration_ms", "subtitle_min_duration_ms"),
        "max_duration_ms": ("max_duration_ms", "maximum_duration_ms", "subtitle_max_duration_ms"),
        "max_cps": ("max_cps", "maximum_cps", "characters_per_second"),
        "max_line_chars": ("max_line_chars", "max_chars_per_line", "maximum_line_length"),
        "max_lines": ("max_lines", "maximum_lines", "max_subtitle_lines"),
        "max_overlap_ms": ("max_overlap_ms", "overlap_tolerance_ms"),
        "asr_chunk_seconds": ("asr_chunk_seconds", "chunk_duration_seconds", "chunk_seconds"),
        "vad_min_silence_seconds": ("vad_min_silence_seconds", "min_silence_duration", "min_silence_seconds"),
        "review_clip_padding_ms": ("review_clip_padding_ms", "audio_review_padding_ms"),
    }
    plausible = {
        "min_duration_ms": (100, 5000),
        "max_duration_ms": (1000, 30000),
        "max_cps": (5, 40),
        "max_line_chars": (20, 80),
        "max_lines": (1, 4),
        "max_overlap_ms": (0, 2000),
        "asr_chunk_seconds": (30, 3600),
        "vad_min_silence_seconds": (0.05, 5),
        "review_clip_padding_ms": (0, 5000),
    }
    candidates: dict[str, list[dict[str, Any]]] = {key: [] for key in defaults}
    structured = _structured_candidates(source_files)
    for target, names in aliases.items():
        low, high = plausible[target]
        for key, value, source in structured:
            leaf = re.split(r"[.\[]", key)[-1].rstrip("]")
            if leaf not in names or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            numeric = float(value)
            if low <= numeric <= high:
                candidates[target].append({"value": value, "source": source, "key": key, "kind": "structured"})
    patterns = {
        "min_duration_ms": re.compile(r"(?i)(?:min(?:imum)?[^\n]{0,30}duration)[^\d]{0,10}(\d{2,5})\s*ms"),
        "max_duration_ms": re.compile(r"(?i)(?:max(?:imum)?[^\n]{0,30}duration)[^\d]{0,10}(\d{3,5})\s*ms"),
        "max_cps": re.compile(r"(?i)(?:max(?:imum)?[^\n]{0,20}cps|characters per second)[^\d]{0,10}(\d{1,2}(?:\.\d+)?)"),
        "max_line_chars": re.compile(r"(?i)(?:max(?:imum)?[^\n]{0,30}(?:characters|chars)[^\n]{0,10}(?:line)?)[^\d]{0,10}(\d{2,3})"),
        "max_lines": re.compile(r"(?i)(?:max(?:imum)?[^\n]{0,20}lines?)[^\d]{0,10}(\d)"),
        "max_overlap_ms": re.compile(r"(?i)(?:overlap[^\n]{0,30})(\d{1,4})\s*ms"),
    }
    for path in source_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for target, pattern in patterns.items():
            low, high = plausible[target]
            for match in pattern.finditer(text):
                value = float(match.group(1))
                if value.is_integer():
                    value = int(value)
                if low <= value <= high:
                    candidates[target].append({"value": value, "source": str(path), "excerpt": match.group(0)[:180], "kind": "text"})
    values = dict(defaults)
    stricter_high = {"min_duration_ms"}
    stricter_low = {"max_duration_ms", "max_cps", "max_line_chars", "max_lines", "max_overlap_ms"}
    for key, rows in candidates.items():
        if not rows:
            candidates[key].append({"value": defaults[key], "source": "v0.1.0 safe fallback", "kind": "fallback"})
            continue
        numbers = [row["value"] for row in rows]
        if key in stricter_high:
            values[key] = max(numbers)
        elif key in stricter_low:
            values[key] = min(numbers)
        else:
            structured_values = [row["value"] for row in rows if row["kind"] == "structured"]
            values[key] = structured_values[0] if structured_values else numbers[0]
    return values, candidates


def _mapping_candidates(source_files: list[Path], category: str) -> list[dict[str, Any]]:
    indicators = ("name", "character", "proper", "cast") if category == "names" else ("gloss", "term", "translation", "reference")
    paths = [path for path in source_files if any(token in path.name.lower() for token in indicators)]
    entries: dict[tuple[str, str], dict[str, Any]] = {}

    def add(source_form: str, canonical: str, source: str) -> None:
        source_form = " ".join(source_form.strip().strip("`*\"'").split())
        canonical = " ".join(canonical.strip().strip("`*\"'").split())
        if not source_form or not canonical or source_form == canonical or len(source_form) > 80 or len(canonical) > 120:
            return
        if any(character in source_form + canonical for character in "{}[]()#"):
            return
        key = (source_form.casefold(), canonical)
        row = entries.setdefault(
            key,
            {"source_form": source_form, "canonical": canonical, "variants": [source_form], "sources": []},
        )
        if source not in row["sources"]:
            row["sources"].append(source)

    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() in {".json", ".yaml", ".yml"}:
            try:
                payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
            except Exception:
                payload = None
            for key, value in _flatten(payload):
                if isinstance(value, str):
                    leaf = re.split(r"[.\[]", key)[-1].rstrip("]")
                    add(leaf, value, str(path))
        for number, line in enumerate(text.splitlines(), 1):
            match = re.match(r"^\s*(?:[-*]\s*)?([^|=→>-]{1,80}?)\s*(?:=>|->|→|\|)\s*([^|]{1,120}?)\s*$", line)
            if match:
                add(match.group(1), match.group(2), f"{path}:{number}")
    return sorted(entries.values(), key=lambda row: (row["canonical"].casefold(), row["source_form"].casefold()))


def _write_config(root: Path, values: dict[str, Any], provenance: dict[str, list[dict[str, Any]]]) -> None:
    payload = {
        "asr": {
            "model": "large-v3",
            "device": "cuda",
            "compute_type": "float16",
            "beam_size": 5,
            "chunk_duration_seconds": int(values["asr_chunk_seconds"]),
            "language": "tr",
            "vad_filter": True,
        },
        "vad": {
            "silence_db": "-35dB",
            "min_silence_seconds": float(values["vad_min_silence_seconds"]),
        },
        "subtitle": {
            "min_duration_ms": int(values["min_duration_ms"]),
            "max_duration_ms": int(values["max_duration_ms"]),
            "max_cps": float(values["max_cps"]),
            "max_line_chars": int(values["max_line_chars"]),
            "max_lines": int(values["max_lines"]),
            "max_overlap_ms": int(values["max_overlap_ms"]),
        },
        "handoff": {"context_blocks": 3},
        "audio_review": {"clip_padding_ms": int(values["review_clip_padding_ms"])},
        "finalize": {"mux_mkv": True},
        "rule_sources": provenance,
    }
    target = root / "config" / "pipeline.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _apply_post_fixes(root: Path) -> None:
    replacements: dict[str, list[tuple[str, str]]] = {
        "src/mas/stages/media.py": [
            ('temp_audio = media_dir / "audio.wav.partial"', 'temp_audio = media_dir / "audio.partial.wav"'),
        ],
        "src/mas/stages/finalize.py": [
            ('temp = target.with_suffix(".mkv.partial")', 'temp = target.with_name(target.stem + ".partial.mkv")'),
            ('"metadata": {"record_count": len(render_rows), "muxed": muxed, "warning_count": len(warnings)},', '"metadata": {"record_count": len(render_rows), "muxed": muxed, "warning_count": len(warnings), "warnings": warnings},'),
        ],
        "src/mas/pipeline.py": [
            (
                'if stage.name == "source":\n        rows: list[dict[str, Any]] = []\n        source_dir = ctx.root / "source"',
                'if stage.name == "source":\n        if ctx.source_url or ctx.fixture:\n            return {"source_url": ctx.source_url, "fixture": ctx.fixture}\n        rows: list[dict[str, Any]] = []\n        source_dir = ctx.root / "source"',
            ),
            (
                '"rules_version": RULES_VERSION,\n            "dependencies": dependencies,',
                '"rules_version": RULES_VERSION,\n            "rules_fingerprint": ctx.settings.rules_fingerprint() if stage.name in {"tr_pack", "tr_return", "id_pack", "id_return", "qc"} else None,\n            "dependencies": dependencies,',
            ),
        ],
    }
    qc_path = root / "src" / "mas" / "stages" / "qc.py"
    text = qc_path.read_text(encoding="utf-8")
    start = text.index("    review_items = [")
    end_marker = "    return {\"outputs\": [json_path, md_path, review_path], \"metadata\": {\"status_result\": status, \"issue_count\": len(issues)}}"
    end = text.index(end_marker, start) + len(end_marker)
    replacement = '    return {"outputs": [json_path, md_path], "metadata": {"status_result": status, "issue_count": len(issues)}}'
    qc_path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
    for relative, rows in replacements.items():
        path = root / relative
        text = path.read_text(encoding="utf-8")
        for old, new in rows:
            if old not in text:
                raise RuntimeError(f"post-fix anchor not found in {relative}: {old[:80]}")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
    mas = root / "mas"
    mas.write_text(
        '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="python3"
fi
exec "$PY" -m mas.cli "$@"
''',
        encoding="utf-8",
    )


def apply(root: Path) -> None:
    root = root.resolve()
    merged = _merge_bootstrap_archive(root)
    all_paths = [*CORE_FILES, *STAGE_FILES, *STATIC_FILES, *TEST_FILES]
    _archive_replaced(root, all_paths)
    manifest = _copy_references(root)
    if manifest["missing_required"]:
        raise RuntimeError(f"required V2 source files are missing: {manifest['missing_required']}")
    inventory, source_files = _rule_inventory(root)
    values, provenance = _select_thresholds(root, source_files)
    names = _mapping_candidates(source_files, "names")
    glossary = _mapping_candidates(source_files, "glossary")
    for mapping in (CORE_FILES, STAGE_FILES, STATIC_FILES, TEST_FILES):
        for relative, content in mapping.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    (root / "rules" / "V2_RULE_INVENTORY.md").write_text(inventory, encoding="utf-8")
    (root / "rules" / "NAMES.json").write_text(
        json.dumps({"schema_version": "1.0", "rules_version": "0.1.0", "entries": names, "generated_from_v2": True}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "rules" / "GLOSSARY.json").write_text(
        json.dumps({"schema_version": "1.0", "rules_version": "0.1.0", "entries": glossary, "generated_from_v2": True}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_config(root, values, provenance)
    _apply_post_fixes(root)
    for executable in (root / "mas", root / "runpod" / "bootstrap.sh", root / "runpod" / "start.sh", root / "runpod" / "preflight.sh"):
        executable.chmod(0o755)
    report = {
        "merged_bootstrap_files": len(merged),
        "required_v2_files": len(manifest["files"]),
        "missing_required": manifest["missing_required"],
        "rule_source_files_scanned": len(source_files),
        "canonical_names": len(names),
        "glossary_entries": len(glossary),
        "selected_thresholds": values,
    }
    (root / "MIGRATION_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    apply(Path.cwd())
