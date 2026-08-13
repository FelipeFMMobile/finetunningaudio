"""Gera a variante LoRA/T4 a partir do notebook didático principal."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "01_qwen3_tts_clone_finetuning.ipynb"
OUTPUT = ROOT / "notebooks" / "01_qwen3_tts_clone_finetuning_loRA.ipynb"


def replace_cell(notebook: dict, marker: str, source: str) -> None:
    matches = [cell for cell in notebook["cells"] if marker in cell.get("source", "")]
    if len(matches) != 1:
        raise RuntimeError(f"Esperava uma célula para {marker!r}; encontrei {len(matches)}.")
    matches[0]["source"] = source.strip() + "\n"
    if matches[0]["cell_type"] == "code":
        matches[0]["execution_count"] = None
        matches[0]["outputs"] = []


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

replace_cell(
    notebook,
    "# Pipeline Qwen3-TTS: da gravação ao fine-tuning",
    """
<a href="https://colab.research.google.com/github/FelipeFMMobile/finetunningaudio/blob/main/notebooks/01_qwen3_tts_clone_finetuning_loRA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir LoRA no Colab"/></a>

# Pipeline Qwen3-TTS LoRA: adaptação leve para T4

Esta variante reaproveita a preparação e o zero-shot do notebook principal, mas substitui o fine-tuning completo por **LoRA (Low-Rank Adaptation)**. Os pesos originais ficam congelados e somente pequenas matrizes são treinadas, reduzindo gradientes e estados do otimizador para caber em uma T4 de 16 GB.

| Parte deste notebook | Referência do curso |
|---|---|
| Parte A — Preparação dos dados | `02_data_prep.ipynb` |
| Parte B — Clonagem zero-shot | `01_voice_cloning.ipynb` |
| Parte C — LoRA e teste | adaptação experimental de `03_finetune.ipynb` |

> LoRA para Qwen3-TTS ainda é uma extensão comunitária, não o caminho oficial básico. Preserve o notebook principal para comparar com full fine-tuning em L4/A100.
""",
)

replace_cell(
    notebook,
    "## 1. Preparação do Colab",
    """
## 1. Preparação do Colab — variante LoRA para T4

Selecione **Ambiente de execução → Alterar tipo de ambiente de execução → GPU T4**. Esta variante usa Qwen3-TTS 0.6B em FP32 porque FP16 é instável na arquitetura Turing da T4. O ganho de memória vem de congelar o modelo-base e treinar apenas os adaptadores LoRA.

Além do Qwen oficial, instalamos PEFT e uma implementação comunitária de LoRA fixada a revisões conhecidas. Isso torna o experimento reproduzível e evita mudanças silenciosas no código externo.
""",
)

replace_cell(
    notebook,
    'QWEN_REPO = Path("/content/Qwen3-TTS")',
    """
QWEN_REPO = Path("/content/Qwen3-TTS")
LORA_REPO = Path("/content/qwen3-tts-lora-finetuning")
QWEN_PIN = "0c6a7cbb6c8421a46332f8c2434c7825c4c855ef"
LORA_PIN = "4076c434e3bc51c928410f28f68a4f76f8f2e715"

if not QWEN_REPO.exists():
    subprocess.run(["git", "clone", "https://github.com/QwenLM/Qwen3-TTS.git", str(QWEN_REPO)], check=True)
    subprocess.run(["git", "-C", str(QWEN_REPO), "checkout", QWEN_PIN], check=True)
if not LORA_REPO.exists():
    subprocess.run(["git", "clone", "https://github.com/instavar/qwen3-tts-lora-finetuning.git", str(LORA_REPO)], check=True)
    subprocess.run(["git", "-C", str(LORA_REPO), "checkout", LORA_PIN], check=True)

QWEN_COMMIT = subprocess.check_output(["git", "-C", str(QWEN_REPO), "rev-parse", "HEAD"], text=True).strip()
if QWEN_COMMIT != QWEN_PIN:
    raise RuntimeError("/content/Qwen3-TTS contém outra revisão. Reinicie a sessão do Colab.")
if subprocess.check_output(["git", "-C", str(LORA_REPO), "rev-parse", "HEAD"], text=True).strip() != LORA_PIN:
    raise RuntimeError("/content/qwen3-tts-lora-finetuning contém outra revisão. Reinicie a sessão.")

lora_patch = LORA_REPO / "patches" / "0001-qwen3-tts-lora.patch"
lora_trainer = QWEN_REPO / "finetuning" / "sft_12hz_lora.py"
if not lora_trainer.exists():
    subprocess.run(["git", "-C", str(QWEN_REPO), "apply", "--check", str(lora_patch)], check=True)
    subprocess.run(["git", "-C", str(QWEN_REPO), "apply", str(lora_patch)], check=True)

subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg", "sox", "libsox-fmt-all"], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", str(QWEN_REPO),
     "peft", "openai-whisper", "pydub", "soundfile", "accelerate", "tensorboard",
     "huggingface_hub", "safetensors"],
    check=True,
)
importlib.invalidate_caches()
if importlib.util.find_spec("qwen_tts") is None or importlib.util.find_spec("peft") is None:
    raise RuntimeError("Qwen3-TTS ou PEFT não ficou disponível. Reinicie a sessão e tente novamente.")
print("Qwen3-TTS:", QWEN_COMMIT)
print("Implementação LoRA:", LORA_PIN)
""",
)

replace_cell(
    notebook,
    "### 1.1 Configuração e entrada dos dados",
    """
### 1.1 Configuração e entrada dos dados

Se você já baixou `*_dataset_preparado.tar.gz`, use `INPUT_MODE = "prepared"` e envie esse pacote nesta mesma T4. Assim, o notebook pula segmentação, Whisper e tokenização. Use `INPUT_MODE = "raw"` somente se estiver começando pela gravação WAV.

Os hiperparâmetros LoRA ficam concentrados nesta célula. Para o primeiro experimento, mantenha rank 8, alpha 16, escala 0,30 e batch 1.
""",
)

config_cell = next(cell for cell in notebook["cells"] if 'SPEAKER_NAME = "felipe"' in cell.get("source", ""))
config_cell["source"] = config_cell["source"].replace(
    'MODEL_SIZE, MIXED_PRECISION, BATCH_SIZE = "0.6B", "no", 1\n    TRAINING_SUPPORTED = False',
    'MODEL_SIZE, MIXED_PRECISION, BATCH_SIZE = "0.6B", "no", 1\n    TRAINING_SUPPORTED = True',
).replace(
    '# use "prepared" ao retomar em uma L4/A100',
    '# use "prepared" para importar o dataset nesta T4',
).replace(
    "RANDOM_SEED = 42\n",
    "RANDOM_SEED = 42\nLORA_RANK = 8\nLORA_ALPHA = 16\nLORA_DROPOUT = 0.05\nLORA_SCALE = 0.30\n",
).replace(
    '"input_mode": INPUT_MODE, "training_supported": TRAINING_SUPPORTED,',
    '"input_mode": INPUT_MODE, "training_supported": TRAINING_SUPPORTED,\n    "adaptation": "lora", "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,',
)

replace_cell(
    notebook,
    "### 2.4 Salvar o dataset antes de trocar a GPU",
    """
### 2.4 Salvar uma cópia do dataset processado

Se `INPUT_MODE = "raw"`, baixe este pacote antes de encerrar a sessão. Ele permite repetir LoRA na T4 ou comparar depois com full fine-tuning em L4/A100 sem executar Whisper novamente. No modo `prepared`, a célula apenas informa que o pacote já foi importado.
""",
)

replace_cell(
    notebook,
    "## 4. Parte C — Fine-tuning e teste",
    """
## 4. Parte C — LoRA e teste (`03_finetune.ipynb`, variante leve)

### 4.1 Preparar e treinar os adaptadores

PEFT injeta matrizes LoRA nas projeções de atenção e MLP (`q/k/v/o`, `gate/up/down`). O Qwen-base permanece congelado. Usamos rank 8, batch 1, FP32 e acumulação de quatro passos na T4. Também aplicamos as correções comunitárias de projeção de texto e alinhamento de labels.

O checkpoint contém o adaptador e o embedding do locutor, não uma cópia completa do modelo. Na inferência, ele deve ser combinado com exatamente o mesmo modelo-base.
""",
)

replace_cell(
    notebook,
    'finetuning_dir = RUN_DIR / "qwen_finetuning"',
    """
finetuning_dir = RUN_DIR / "qwen_finetuning_lora"
shutil.copytree(QWEN_REPO / "finetuning", finetuning_dir, dirs_exist_ok=True)
train_script = finetuning_dir / "sft_12hz_lora.py"
source = train_script.read_text(encoding="utf-8")

old_accelerator = '''accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="tensorboard",
    )'''
new_accelerator = '''logging_dir = os.environ.get("QWEN_LOGGING_DIR", "./logs")
    os.makedirs(logging_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir,
    )'''
if old_accelerator not in source:
    raise RuntimeError("A implementação LoRA mudou na configuração do Accelerate.")
source = source.replace(old_accelerator, new_accelerator, 1)

old_load = '''qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )'''
new_load = '''model_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        dtype=model_dtype,
        attn_implementation=args.attn_implementation,
    )'''
if old_load not in source:
    raise RuntimeError("A implementação LoRA mudou no carregamento do modelo.")
source = source.replace(old_load, new_load, 1)
source = source.replace(
    "optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)",
    "optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, foreach=False)",
    1,
)
train_script.write_text(source, encoding="utf-8")
print("Treinador LoRA preparado:", train_script)
""",
)

replace_cell(
    notebook,
    'train_log = RUN_DIR / "train.log"',
    """
train_log = RUN_DIR / "train_lora.log"
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["QWEN_LOGGING_DIR"] = str(RUN_DIR / "logs_lora")
command = [
    sys.executable, str(train_script),
    "--init_model_path", str(MODEL_PATH),
    "--output_model_path", str(RUN_DIR / "adapters"),
    "--train_jsonl", str(train_codes),
    "--batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--mixed_precision", MIXED_PRECISION,
    "--attn_implementation", "sdpa",
    "--lr", str(LEARNING_RATE),
    "--num_epochs", str(NUM_EPOCHS),
    "--speaker_name", SPEAKER_NAME,
    "--lora_rank", str(LORA_RANK),
    "--lora_alpha", str(LORA_ALPHA),
    "--lora_dropout", str(LORA_DROPOUT),
]

print("Iniciando LoRA. O modelo-base ficará congelado.")
with train_log.open("w", encoding="utf-8") as log_handle:
    process = subprocess.Popen(command, cwd=finetuning_dir, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
    return_code = process.wait()
if return_code != 0:
    raise RuntimeError("Treinamento LoRA falhou. Consulte train_lora.log.")

checkpoints = sorted((RUN_DIR / "adapters").glob("checkpoint-epoch-*"),
                     key=lambda path: int(path.name.rsplit("-", 1)[-1]))
if not checkpoints:
    raise RuntimeError("Nenhum adaptador LoRA foi criado.")
print("Adaptadores:", [path.name for path in checkpoints])
""",
)

replace_cell(
    notebook,
    "### 4.2 Carregar o checkpoint e comparar",
    """
### 4.2 Carregar o adaptador e comparar

O adaptador será aplicado ao mesmo Qwen3-TTS 0.6B. Começamos com escala LoRA 0,30, pois escala 1,0 pode exagerar a adaptação. Geramos as mesmas frases do zero-shot para comparar identidade, inteligibilidade, naturalidade e estabilidade. Depois, vale repetir com 0,20, 0,35 e 0,50.
""",
)

replace_cell(
    notebook,
    'if not checkpoints:\n    raise FileNotFoundError("Nenhum checkpoint foi criado.")',
    """
selected_checkpoint = checkpoints[-1]
infer_script = finetuning_dir / "infer_lora_custom_voice.py"
dtype_name = "fp32" if MIXED_PRECISION == "no" else MIXED_PRECISION

for index, sentence in enumerate(TEST_SENTENCES, start=1):
    output = RUN_DIR / "samples" / f"lora_{index}.wav"
    command = [
        sys.executable, str(infer_script),
        "--base_model_path", str(MODEL_PATH),
        "--adapter_path", str(selected_checkpoint),
        "--speaker_name", SPEAKER_NAME,
        "--text", sentence,
        "--language", LANGUAGE,
        "--output_wav", str(output),
        "--dtype", dtype_name,
        "--attn_implementation", "sdpa",
        "--lora_scale", str(LORA_SCALE),
        "--seed", str(RANDOM_SEED),
    ]
    subprocess.run(command, cwd=finetuning_dir, check=True)
    print(f"Frase {index} — zero-shot")
    display(Audio(filename=str(RUN_DIR / "samples" / f"zero_shot_{index}.wav")))
    print(f"Frase {index} — LoRA (escala {LORA_SCALE:.2f})")
    display(Audio(filename=str(output)))
""",
)

notebook["metadata"]["colab"]["name"] = OUTPUT.name
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Notebook LoRA gerado: {OUTPUT} ({len(notebook['cells'])} células)")
