from __future__ import annotations

from pathlib import Path

path = Path("src/mas/pipeline.py")
text = path.read_text(encoding="utf-8")

if "config_hash," in text and "config_hash =" not in text:
    marker = "        state = load(state_path, episode)\n"
    if marker in text:
        text = text.replace(
            marker,
            marker + "        config_hash = rules_fingerprint()\n",
            1,
        )

# A checkpoint must not invalidate itself merely because its own generated source file now exists.
if 'source_input = sha256_json({"fixture": True})' not in text:
    old = '''        source_input = sha256_json(
            {
                "source_url": source_url or "",
                "fixture": fixture,
                "existing": [
                    (path.name, sha256_file(path))
                    for path in sorted((directory / "source").glob("*"))
                    if path.is_file()
                ],
            }
        )
'''
    new = '''        if fixture:
            source_input = sha256_json({"fixture": True})
        elif source_url is None and source_marker.exists():
            previous_source = read_json(source_marker)
            previous_path = Path(previous_source["path"])
            source_input = sha256_json(
                {
                    "source_url": previous_source.get("source_url") or "",
                    "source_sha256": (
                        sha256_file(previous_path) if previous_path.exists() else "missing"
                    ),
                }
            )
        else:
            source_input = sha256_json(
                {
                    "source_url": source_url or "",
                    "existing": [
                        (path.name, sha256_file(path))
                        for path in sorted((directory / "source").glob("*"))
                        if path.is_file()
                    ],
                }
            )
'''
    if old in text:
        text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print("Applied defensive checkpoint compatibility fix.")
