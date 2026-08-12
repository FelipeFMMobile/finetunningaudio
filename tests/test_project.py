import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_qwen3_tts_clone_finetuning.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_notebook_has_all_didactic_sections_in_order() -> None:
    notebook = load_notebook()
    markdown = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    expected_titles = [
        "## 1. Preparação do Colab",
        "## 2. Parte A — Preparação dos dados",
        "## 3. Parte B — Clonagem zero-shot",
        "## 4. Parte C — Fine-tuning e teste",
        "## 5. Exportar os resultados para o Mac",
        "## Apêndice — retomada de uma execução interrompida",
    ]
    positions = [markdown.index(title) for title in expected_titles]
    assert positions == sorted(positions)
    for reference in ["01_voice_cloning.ipynb", "02_data_prep.ipynb", "03_finetune.ipynb"]:
        assert reference in markdown


def test_all_python_cells_compile_and_have_no_saved_outputs() -> None:
    notebook = load_notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        ast.parse(cell["source"], filename=f"cell_{index}.py")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_notebook_targets_colab_gpu_and_local_transfer() -> None:
    notebook = load_notebook()
    source = "\n".join(cell["source"] for cell in notebook["cells"])
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert "files.upload()" in source
    assert "files.download(str(export_path))" in source
    assert 'drive.mount("/content/drive")' not in source
    assert "Qwen3-TTS-12Hz-{MODEL_SIZE}-Base" in source
    assert '"QWEN_SYNC_DIR"' not in source
    assert "avaliacao_auditiva.csv" not in source
    assert 'find_spec("qwen_tts")' in source
    assert '"-e", str(QWEN_REPO)' not in source
    assert '"sox", "libsox-fmt-all"' in source


def test_recording_script_has_expected_size_and_segments() -> None:
    script = (ROOT / "scripts" / "roteiro_gravacao_ptbr.txt").read_text(encoding="utf-8")
    words = re.findall(r"\b[\wÀ-ÿ-]+\b", script)
    assert 900 <= len(words) <= 1200
    assert len(re.findall(r"\[S\d{2}\]", script)) == 24


def test_gitignore_blocks_private_artifacts() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ["*.wav", "*.safetensors", "*.jsonl", ".env", "checkpoints/"]:
        assert pattern in gitignore
