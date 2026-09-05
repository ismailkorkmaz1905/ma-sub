from __future__ import annotations

import json
from pathlib import Path
import unittest

from mas.engine.tr_correction import DEFAULT_INSTRUCTIONS


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2] / "legacy" / "SYSTEM_V2_BETA"
)


class V2InstructionsContractTests(unittest.TestCase):
    def test_embedded_tr_instructions_state_the_p0_non_dialogue_gate(self) -> None:
        instructions = " ".join(DEFAULT_INSTRUCTIONS.split())
        self.assertIn(
            "Every non-flagged ASR record is dialogue", instructions
        )
        self.assertIn("always `false` in this text-only pass", instructions)
        self.assertIn("`suspected_asr_hallucination`", instructions)
        self.assertIn("`orphan_youtube_caption`", instructions)
        self.assertIn("always `false` in this text-only pass", instructions)
        self.assertIn("`_TR_TEXT_CORRECTED.zip`", instructions)
        self.assertIn("The next Colab stage owns all audio decisions", instructions)
        self.assertIn(
            "Every non-flagged ASR record is dialogue and requires non-empty",
            instructions,
        )
        self.assertIn("EXTRA_AUDIO_REVIEW_UIDS", instructions)
        self.assertIn("Do not browse or search the web", instructions)
        self.assertIn("Do not search Hugging Face, GitHub", instructions)
        self.assertIn("Do not open, inspect, transcribe or listen", instructions)
        self.assertIn(
            "only permitted external connector is Google Drive", instructions
        )
        self.assertIn("model/tool discovery are forbidden", instructions)

    def test_project_instructions_and_episode_prompt_repeat_the_gate(self) -> None:
        project = (PROJECT_ROOT / "V2_PROJECT_INSTRUCTIONS.md").read_text(
            encoding="utf-8"
        )
        prompts = (PROJECT_ROOT / "V2_NEW_EPISODE_PROMPTS.md").read_text(
            encoding="utf-8"
        )
        project_flat = " ".join(project.split())
        prompts_flat = " ".join(prompts.split())
        self.assertIn(
            "Every ordinary record without an immutable audio-review entry", project
        )
        self.assertIn("This pass is text-only", project)
        self.assertIn("`review_disposition=pending_audio_review`", project)
        self.assertIn("TR_TEXT_CORRECTED.zip", project)
        self.assertIn("SYSTEM_V2_BETA", project)
        self.assertIn(
            "Immutable audio-review kaydı olmayan normal ASR", prompts
        )
        self.assertIn("Bu aşama yalnız metin düzeltmesidir", prompts)
        self.assertIn("WAV dosyalarını açma", prompts)
        self.assertIn("TR_TEXT_CORRECTED.zip", prompts)
        self.assertIn("EXTRA_AUDIO_REVIEW_UIDS", prompts)
        self.assertIn("Never browse or search the web", project)
        self.assertIn("do not download or install Whisper", project)
        self.assertIn("Translate with the model itself", project)
        self.assertIn("Web araması yapma", prompts)
        self.assertIn("ASR modeli indirme/kurma", prompts)
        self.assertIn("İzin verilen tek harici connector", prompts)
        self.assertIn("model/tool keşfi yasaktır", prompts)
        self.assertIn("only permitted external connector is Google Drive", project)
        self.assertIn("Çeviriyi kendin yap", prompts)
        for style_rule in (
            "spoken dialogue, not literal machine translation",
            "`aku`, `kamu`, `nggak`, `udah` and `aja`",
            "`saya`, `Anda`, `Pak` and `Bu`",
            "romance, anger, sarcasm, comedy",
        ):
            self.assertIn(style_rule, project_flat)
        for style_rule in (
            "kelime kelime makine çevirisi yapma",
            "aku, kamu, nggak, udah ve aja",
            "saya, Anda, Pak ve Bu",
            "romantizmi, öfkeyi, alayı, mizahı",
        ):
            self.assertIn(style_rule, prompts_flat)

    def test_prepare_notebook_warns_before_creating_the_pack(self) -> None:
        notebook = json.loads(
            (PROJECT_ROOT / "01_PREPARE_TR.ipynb").read_text(encoding="utf-8")
        )
        markdown = "\n".join(
            cell.get("source", "")
            if isinstance(cell.get("source", ""), str)
            else "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        self.assertIn("ordinary non-flagged ASR record remains dialogue", markdown)
        self.assertIn("`non_dialogue=false`", markdown)
        self.assertIn("`review_disposition=pending_audio_review`", markdown)
        self.assertIn("Colab, not the text model", markdown)


if __name__ == "__main__":
    unittest.main()
