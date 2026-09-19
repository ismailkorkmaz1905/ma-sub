import json
import zipfile

import pytest

from mas import pipeline


def words():
    return [
        {"text": "Merhaba", "start_ms": 100, "end_ms": 350},
        {"text": "dünya.", "start_ms": 400, "end_ms": 700, "break_after": True},
        {"text": "Nasılsın?", "start_ms": 1900, "end_ms": 2400, "break_after": True},
    ]


def blocks():
    return [
        {
            "block_uid": "MA15-00001-aaaaaaaa",
            "start_ms": 100,
            "end_ms": 900,
            "source_text": "Merhaba dünya.",
        },
        {
            "block_uid": "MA15-00002-bbbbbbbb",
            "start_ms": 1900,
            "end_ms": 2600,
            "source_text": "Nasılsın?",
        },
    ]


def translated(source_blocks):
    return [
        {"block_uid": source_blocks[0]["block_uid"], "tr": "Merhaba dünya.", "id": "Halo dunia."},
        {"block_uid": source_blocks[1]["block_uid"], "tr": "Nasılsın?", "id": "Apa kabar?"},
    ]


def write_return(path, manifest, records):
    return_manifest = {
        "format": "mas-translated-1",
        "pack_id": manifest["pack_id"],
        "block_count": manifest["block_count"],
    }
    pipeline._atomic_bytes(
        path,
        pipeline._zip_bytes(
            {
                "manifest.json": json.dumps(return_manifest).encode(),
                "translated.jsonl": pipeline._jsonl_bytes(records),
            }
        ),
    )


def test_build_blocks_is_deterministic_and_non_overlapping():
    config = pipeline._load_config()["subtitle"]
    first = pipeline.build_blocks(15, "a" * 64, words(), config)
    second = pipeline.build_blocks(15, "a" * 64, words(), config)
    assert first == second
    assert [item["source_text"] for item in first] == ["Merhaba dünya.", "Nasılsın?"]
    assert first[0]["end_ms"] <= first[1]["start_ms"]


def test_build_blocks_removes_exact_credit_metadata():
    config = pipeline._load_config()["subtitle"]
    source = [
        {"text": "Altyazı", "start_ms": 0, "end_ms": 200},
        {"text": "M.K.", "start_ms": 220, "end_ms": 500, "break_after": True},
        {"text": "Gerçek", "start_ms": 1500, "end_ms": 1800},
        {"text": "replik.", "start_ms": 1820, "end_ms": 2200, "break_after": True},
    ]
    result = pipeline.build_blocks(15, "b" * 64, source, config)
    assert [item["source_text"] for item in result] == ["Gerçek replik."]


def test_pack_is_small_and_deterministic(tmp_path):
    path = tmp_path / "pack.zip"
    config = pipeline._load_config()
    first = pipeline._create_pack(path, 15, "c" * 64, blocks(), config)
    first_bytes = path.read_bytes()
    second = pipeline._create_pack(path, 15, "c" * 64, blocks(), config)
    assert first == second
    assert path.read_bytes() == first_bytes
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {
            "INSTRUCTIONS.md",
            "blocks.jsonl",
            "glossary.json",
            "manifest.json",
        }


def test_translation_validation_accepts_text_only_return(tmp_path):
    config = pipeline._load_config()
    source_blocks = blocks()
    pack = pipeline._create_pack(tmp_path / "pack.zip", 15, "d" * 64, source_blocks, config)
    result_path = tmp_path / "translated.zip"
    expected = translated(source_blocks)
    write_return(result_path, pack, expected)
    assert pipeline._validate_translation(result_path, pack, source_blocks, config) == expected


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda records: records.reverse(), "order or identity"),
        (lambda records: records[0].update({"start_ms": 100}), "unexpected fields"),
        (lambda records: records.pop(), "block count"),
    ],
)
def test_translation_validation_fails_closed(tmp_path, change, error):
    config = pipeline._load_config()
    source_blocks = blocks()
    pack = pipeline._create_pack(tmp_path / "pack.zip", 15, "e" * 64, source_blocks, config)
    records = translated(source_blocks)
    change(records)
    result_path = tmp_path / "translated.zip"
    write_return(result_path, pack, records)
    with pytest.raises(RuntimeError, match=error):
        pipeline._validate_translation(result_path, pack, source_blocks, config)


def test_translation_validation_rejects_stale_pack(tmp_path):
    config = pipeline._load_config()
    source_blocks = blocks()
    pack = pipeline._create_pack(tmp_path / "pack.zip", 15, "f" * 64, source_blocks, config)
    result_path = tmp_path / "translated.zip"
    stale = dict(pack)
    stale["pack_id"] = "0" * 64
    write_return(result_path, stale, translated(source_blocks))
    with pytest.raises(RuntimeError, match="stale"):
        pipeline._validate_translation(result_path, pack, source_blocks, config)


def test_religious_translation_is_checked(tmp_path):
    config = pipeline._load_config()
    source_blocks = blocks()[:1]
    pack = pipeline._create_pack(tmp_path / "pack.zip", 15, "1" * 64, source_blocks, config)
    records = [{"block_uid": source_blocks[0]["block_uid"], "tr": "Allah aşkına yapma.", "id": "Tolong jangan."}]
    result_path = tmp_path / "translated.zip"
    write_return(result_path, pack, records)
    with pytest.raises(RuntimeError, match="does not preserve"):
        pipeline._validate_translation(result_path, pack, source_blocks, config)


def test_srt_has_two_lines_at_most(tmp_path):
    source_blocks = blocks()[:1]
    records = [
        {
            "block_uid": source_blocks[0]["block_uid"],
            "tr": "Bu satır dengeli biçimde iki kısa altyazı satırına bölünecek.",
            "id": "Kalimat ini akan dibagi menjadi dua baris subtitle yang pendek.",
        }
    ]
    target = tmp_path / "test.srt"
    pipeline._write_srt(target, source_blocks, records, "tr", 42)
    body = target.read_text(encoding="utf-8").strip().splitlines()
    assert body[1] == "00:00:00,100 --> 00:00:00,900"
    assert len(body[2:]) <= 2
    assert all(len(line) <= 42 for line in body[2:])


def test_run_stops_for_one_handoff_then_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "EPISODES_ROOT", tmp_path / "episodes")
    monkeypatch.setattr(pipeline, "_probe_video", lambda path: None)
    monkeypatch.setattr(pipeline, "_transcribe_source", lambda source, config: words())

    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video")
    assert pipeline.run(15, source_file=source, subtitles_only=True) == pipeline.WAIT_TRANSLATION

    paths = pipeline._paths(15)
    state = pipeline.status(15)
    source_blocks = pipeline._read_jsonl(paths["work"] / "blocks.jsonl")
    pack_manifest = json.loads(
        zipfile.ZipFile(paths["root"] / state["pack"]["path"]).read("manifest.json")
    )
    output = translated(source_blocks)
    write_return(
        paths["handoff"] / f"{paths['root'].name}_TRANSLATED.zip",
        pack_manifest,
        output,
    )

    assert pipeline.run(15, subtitles_only=True) == 0
    assert pipeline.status(15)["stage"] == "DONE_SUBTITLES"
    assert (paths["output"] / f"{paths['root'].name}.tr.srt").is_file()
    assert (paths["output"] / f"{paths['root'].name}.id.srt").is_file()


def test_recorded_source_is_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "EPISODES_ROOT", tmp_path / "episodes")
    monkeypatch.setattr(pipeline, "_probe_video", lambda path: None)
    monkeypatch.setattr(pipeline, "_transcribe_source", lambda source, config: words())
    source = tmp_path / "source.mp4"
    source.write_bytes(b"first")
    pipeline.run(15, source_file=source, subtitles_only=True)
    recorded = pipeline._paths(15)["source"] / "video.mp4"
    recorded.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256 changed"):
        pipeline.run(15, subtitles_only=True)
