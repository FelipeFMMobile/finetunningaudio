import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_qwen3_tts_clone_finetuning.ipynb"
LORA_NOTEBOOK_PATH = ROOT / "notebooks" / "01_qwen3_tts_clone_finetuning_loRA.ipynb"


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
    for path in [NOTEBOOK_PATH, LORA_NOTEBOOK_PATH]:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells
        for index, cell in enumerate(code_cells):
            ast.parse(cell["source"], filename=f"{path.name}:cell_{index}.py")
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
    assert 'MODEL_SIZE, MIXED_PRECISION, BATCH_SIZE = "0.6B", "no", 1' in source
    assert 'project_dir=logging_dir' in source
    assert 'env["QWEN_LOGGING_DIR"] = str(RUN_DIR / "logs")' in source
    assert '"no": torch.float32' in source
    assert 'model.talker.text_projection(input_text_embedding)' in source
    assert 'dtype=model_dtype' in source
    assert 'foreach=False if low_memory else None' in source
    assert 'env["QWEN_LOW_MEMORY"] = "1" if GPU_MEMORY_GB < 20 else "0"' in source
    assert 'INPUT_MODE = "raw"' in source
    assert 'INPUT_MODE == "prepared"' in source
    assert 'TRAINING_SUPPORTED = False' in source
    assert "Fine-tuning completo bloqueado nesta GPU" in source
    assert "_dataset_preparado.tar.gz" in source
    assert 'archive.extractall(RUN_DIR, filter="data")' in source


def test_recording_script_has_expected_size_and_segments() -> None:
    script = (ROOT / "scripts" / "roteiro_gravacao_ptbr.txt").read_text(encoding="utf-8")
    words = re.findall(r"\b[\wÀ-ÿ-]+\b", script)
    assert 900 <= len(words) <= 1200
    assert len(re.findall(r"\[S\d{2}\]", script)) == 24


def test_lora_notebook_is_separate_and_t4_oriented() -> None:
    notebook = json.loads(LORA_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "Low-Rank Adaptation" in source
    assert 'LORA_RANK = 8' in source
    assert 'LORA_SCALE = 0.30' in source
    assert 'MODEL_SIZE, MIXED_PRECISION, BATCH_SIZE = "0.6B", "no", 1' in source
    assert 'TRAINING_SUPPORTED = True' in source
    assert "sft_12hz_lora.py" in source
    assert "infer_lora_custom_voice.py" in source
    assert '"peft"' in source
    assert 'infer_source.rstrip().endswith(main_guard)' in source
    assert 'compile(infer_lora_script.read_text' in source


def test_gitignore_blocks_private_artifacts() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ["*.wav", "*.safetensors", "*.jsonl", ".env", "checkpoints/"]:
        assert pattern in gitignore
