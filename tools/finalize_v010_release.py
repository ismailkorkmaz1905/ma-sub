from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path


def write(path: str, text: str, executable: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    if executable:
        target.chmod(0o755)


pipeline_path = Path("src/mas/pipeline.py")
if not pipeline_path.exists():
    raise SystemExit("src/mas/pipeline.py is missing")
pipeline = pipeline_path.read_text(encoding="utf-8")

# The source checkpoint must be keyed by the acquired media bytes and canonical
# source URL. Generated files in source/ must never change the key on resume.
start_marker = '        source_marker = directory / "work" / "source.json"\n'
end_marker = '        if stop_after == "source":\n'
if start_marker in pipeline and end_marker in pipeline:
    start = pipeline.index(start_marker)
    end = pipeline.index(end_marker, start)
    replacement = '''        source_marker = directory / "work" / "source.json"
        previous_source = read_json(source_marker) if source_marker.exists() else {}
        canonical_url = source_url or previous_source.get("source_url") or ""
        previous_path_value = previous_source.get("path")
        previous_path = Path(previous_path_value) if previous_path_value else None
        if (
            source_url is None
            and previous_path is not None
            and previous_path.exists()
        ):
            source = previous_path
        else:
            source = acquire_source(directory, source_url, fixture)
        source_input = sha256_json(
            {
                "source_url": canonical_url,
                "source_sha256": sha256_file(source),
                "fixture": fixture,
            }
        )

        def source_action() -> dict:
            write_json(
                source_marker,
                {
                    "path": str(source),
                    "sha256": sha256_file(source),
                    "source_url": canonical_url or None,
                },
            )
            return {"path": str(source), "source_url": canonical_url or None}

        run_checkpoint(
            state,
            state_path,
            "source",
            source_input,
            stage_config_hash("source") if "stage_config_hash" in globals() else config_hash,
            [source_marker],
            source_action,
        )
        source = Path(read_json(source_marker)["path"])
'''
    pipeline = pipeline[:start] + replacement + pipeline[end:]

# Honor an explicitly changed URL without deleting the previous source media.
acquire_start = '    def acquire_source(directory: Path, source_url: str | None, fixture: bool) -> Path:\n'
asr_start = '    def run_asr(audio: Path, output: Path, fixture: bool) -> dict:\n'
if acquire_start in pipeline and asr_start in pipeline:
    start = pipeline.index(acquire_start)
    end = pipeline.index(asr_start, start)
    replacement = r'''    def acquire_source(directory: Path, source_url: str | None, fixture: bool) -> Path:
        source_directory = directory / "source"
        source_directory.mkdir(parents=True, exist_ok=True)
        if fixture:
            target = source_directory / "fixture.source"
            target.write_text("offline fixture\n", encoding="utf-8")
            return target
        if source_url:
            source_key = sha256_json(source_url)[:12]
            local = Path(
                source_url[7:] if source_url.startswith("file://") else source_url
            ).expanduser()
            if local.exists():
                suffix = local.suffix or ".mkv"
                target = source_directory / f"source-{source_key}{suffix}"
                if not target.exists() or sha256_file(target) != sha256_file(local):
                    shutil.copy2(local, target)
                return target
            if not shutil.which("yt-dlp"):
                raise RuntimeError("yt-dlp is required for URL acquisition")
            template = source_directory / f"source-{source_key}.%(ext)s"
            existing = sorted(source_directory.glob(f"source-{source_key}.*"))
            media_existing = [
                path
                for path in existing
                if path.suffix.lower() in {".mkv", ".mp4", ".webm", ".mov"}
            ]
            if media_existing:
                return media_existing[0]
            subprocess.run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--merge-output-format",
                    "mkv",
                    "--write-subs",
                    "--write-auto-subs",
                    "--sub-langs",
                    "tr,tr-TR",
                    "--sub-format",
                    "vtt",
                    "-o",
                    str(template),
                    source_url,
                ],
                check=True,
            )
            candidates = sorted(
                path
                for path in source_directory.glob(f"source-{source_key}.*")
                if path.suffix.lower() in {".mkv", ".mp4", ".webm", ".mov"}
            )
            if not candidates:
                raise RuntimeError("yt-dlp completed without a source media file")
            return candidates[0]
        existing = sorted(
            path
            for path in source_directory.iterdir()
            if path.suffix.lower() in {".mkv", ".mp4", ".webm", ".mov"}
        )
        if existing:
            return existing[0]
        raise RuntimeError(
            "No source media. Pass --source-url once or place a video under source/."
        )


'''
    pipeline = pipeline[:start] + replacement + pipeline[end:]

# Use timestamped traceback files instead of overwriting the prior failure.
cli_path = Path("src/mas/cli.py")
if cli_path.exists():
    cli = cli_path.read_text(encoding="utf-8")
    if "from datetime import datetime, timezone" not in cli:
        cli = cli.replace(
            "import traceback\n",
            "import traceback\nfrom datetime import datetime, timezone\n",
        )
    cli = cli.replace(
        '            traceback_path = logs / "last_failure.traceback.log"\n',
        '            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")\n'
        '            traceback_path = logs / f"{stamp}_{arguments.command}.traceback.log"\n',
    )
    cli_path.write_text(cli, encoding="utf-8")

pipeline_path.write_text(pipeline, encoding="utf-8")

# Append a real-source checkpoint regression without changing the fixture test.
test_path = Path("tests/release/test_release_runtime.py")
if test_path.exists():
    tests = test_path.read_text(encoding="utf-8")
    marker = "def test_real_source_checkpoint_is_stable_without_repeating_url"
    if marker not in tests:
        tests += '''


def test_real_source_checkpoint_is_stable_without_repeating_url(tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_WORKSPACE", str(tmp_path / "workspace"))
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"stable-media-bytes")
    episode = 99
    assert pipeline.run(episode, source_url=str(source), stop_after="source") == 0
    directory = (
        tmp_path
        / "workspace"
        / "EPISODES"
        / "Muhtemel Ask 99.Bolum"
    )
    first = json.loads((directory / "work" / "state.json").read_text())
    completed_at = first["stages"]["source"]["completed_at"]
    assert pipeline.run(episode, stop_after="source") == 0
    second = json.loads((directory / "work" / "state.json").read_text())
    assert second["stages"]["source"]["completed_at"] == completed_at
'''
        test_path.write_text(tests, encoding="utf-8")

# Operator-first documentation required for weekly use.
write(
    "README.md",
    '''
    # ma-sub

    Production-oriented, restart-safe Turkish to Indonesian subtitle pipeline for
    "Muhtemel Ask". Notebooks are retained only as legacy/reference material and are
    not production entry points.

    ## What do I do when episode 13 comes out?

    1. Start an NVIDIA RunPod with a persistent volume mounted at `/workspace`.
    2. Clone this private repository and check out the stable tag.
    3. Run `./runpod/bootstrap.sh` once for that persistent environment.
    4. Run `./mas doctor`.
    5. Run `./mas run 13 --source-url "SOURCE_URL"`.
    6. When the console says `GPU WORK COMPLETE`, stop the GPU pod.
    7. Give `translation_input/TR_PROMPT.txt` and the generated TR ZIP to ChatGPT Pro.
    8. Put the exact returned ZIP in `translation_output/`, then run `./mas run 13`.
    9. Repeat the generated ID handoff when requested, then run `./mas run 13` again.
    10. Read the Indonesian SRT under `final/subtitles/` and QC under `reports/`.

    The same `run` command validates checkpoints and continues from the first invalid or
    incomplete stage. Do not delete `work/state.json` to retry a normal failure.

    ## First-time RunPod setup

    ```bash
    git clone https://github.com/ismailkorkmaz1905/ma-sub.git /workspace/ma-sub
    cd /workspace/ma-sub
    git checkout v0.1.0
    ./runpod/bootstrap.sh
    ./mas doctor
    ```

    For a private HTTPS clone, configure a GitHub token once in the RunPod environment
    or use an SSH deploy key. Set `MAS_WORKSPACE=/workspace/ma-sub` only when the
    repository is not already on the persistent volume.

    ## Commands

    ```bash
    ./mas run 13 --source-url "SOURCE_URL"
    ./mas run 13
    ./mas status 13
    ./mas doctor
    ./mas test
    ./mas clean 13 --dry-run
    ./mas clean 13 --destroy
    ```

    `clean` never removes source media or returned translation packages. Destructive
    cleanup requires `--destroy` and is limited to regenerable caches.

    ## Recovery after an error

    The concise failure is printed to the console, while the full traceback is written
    to the episode `logs/` directory. Fix the reported cause and rerun:

    ```bash
    ./mas run 13
    ```

    A valid completed ASR checkpoint is not recomputed when a later translation,
    alignment, finalization, or QC stage changes.

    ## Updating and rollback

    Update only after the current episode output is safe:

    ```bash
    git fetch --tags origin
    git checkout main
    git pull --ff-only
    ./mas test
    ```

    Roll back to the last stable release:

    ```bash
    git fetch --tags origin
    git checkout v0.1.0
    ```

    Every episode state records the pipeline, schema, and rules versions plus input and
    output SHA-256 values. Returned ChatGPT ZIPs are rejected when episode, schema,
    schema hash, input hash, count, order, UID, index, timing, or another immutable
    field changes.

    ## v0.1.0 validation boundary

    The release gate covers the complete CPU regression suite, CLI, checkpoint/resume,
    both structured ChatGPT handoffs, strict invariants, SRT rendering, QC generation,
    and a Docker smoke build. A real full episode GPU inference run is deliberately not
    claimed until it is run on RunPod.
    ''',
)

write(
    "runpod/README.md",
    '''
    # RunPod

    Use a persistent `/workspace` volume. Clone the repository there, check out the
    stable tag, then run:

    ```bash
    cd /workspace/ma-sub
    ./runpod/bootstrap.sh
    ./mas doctor
    ./mas run 13 --source-url "SOURCE_URL"
    ```

    `bootstrap.sh` creates `.venv` and installs the exact lock file. Model downloads and
    episode artifacts remain on the persistent volume. The pipeline clearly prints when
    GPU work is complete and it is safe to stop the pod.
    ''',
)

write(
    "CHANGELOG.md",
    '''
    # Changelog

    ## 0.1.0 - 2026-09-04

    - Replaced notebook execution with one restart-safe CLI.
    - Added SHA-256 and version-bound per-stage checkpoints.
    - Added explicit ChatGPT Pro TR correction and ID translation handoffs.
    - Added strict returned-package invariant validation.
    - Added SRT rendering, QC reports, structured stage logs, and safe cleanup.
    - Preserved the V2 source, rules, notebooks, and regression suite as reference.
    - Added Docker and RunPod startup paths.
    ''',
)

write(
    "MIGRATION_NOTES.md",
    '''
    # V2 migration notes

    Source inspected: `Muhtemel_Ask_Subtitle_System_V2_BETA_2026-09-04.zip` and the
    `SYSTEM_V2_BETA` Drive folder, including the four V2 notebooks, project/readme and
    translation instructions, config, Python source, test reports, requirements, and
    tests.

    Existing deterministic V2 logic and tests were retained rather than redesigned.
    Notebook-only orchestration was moved behind the `mas` CLI. Immutable fields,
    schema/hash binding, audio-review flags, alignment provenance, Turkish correction,
    Indonesian translation, archive safety, and subtitle constraints remain versioned.
    Where documentation conflicts, the stricter invariant applies unless the source
    documentation declares precedence.

    Notebooks under legacy/reference are historical material only. Production execution
    must not import Google Colab.
    ''',
)

# A simple guard ensures release metadata and lock entries do not silently float.
lock_path = Path("requirements.lock")
if lock_path.exists():
    unpinned = []
    for raw in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if "==" not in line and " @ " not in line:
            unpinned.append(line)
    if unpinned:
        raise SystemExit(f"Unpinned requirements.lock entries: {unpinned}")

print("Final v0.1.0 release patch applied.")
