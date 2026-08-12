"""Gera o notebook didático e compacto sem depender de nbformat."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_qwen3_tts_clone_finetuning.ipynb"
cells: list[dict] = []


def md(source: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n",
        }
    )


def code(source: str, *, compact: bool = False) -> None:
    metadata = {"cellView": "form"} if compact else {}
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": metadata,
            "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n",
        }
    )


md(
    """
    <a href="https://colab.research.google.com/github/FelipeFMMobile/finetunningaudio/blob/main/notebooks/01_qwen3_tts_clone_finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir no Colab"/></a>

    # Pipeline Qwen3-TTS: da gravação ao fine-tuning

    Este notebook reúne o fluxo completo do curso em um único experimento para Google Colab. A ordem foi ajustada porque extrairemos o áudio de referência da própria gravação: primeiro preparamos os dados, depois criamos a linha de base zero-shot e, por último, treinamos o modelo.

    | Parte deste notebook | Referência do curso |
    |---|---|
    | Parte A — Preparação dos dados | `02_data_prep.ipynb` |
    | Parte B — Clonagem zero-shot | `01_voice_cloning.ipynb` |
    | Parte C — Fine-tuning e teste | `03_finetune.ipynb` |

    > Use somente sua própria voz ou uma voz com consentimento explícito. Áudio e pesos não serão enviados ao GitHub.
    """
)

md(
    """
    ## 1. Preparação do Colab — equivalente ao “Setup” dos três notebooks

    O navegador está no seu Mac, mas o código roda em uma máquina virtual do Google. Selecione **Ambiente de execução → Alterar tipo de ambiente de execução → GPU**. Uma T4 usará o Qwen3-TTS 0.6B em FP16; L4 e A100 usarão o 1.7B em BF16.

    A primeira célula valida CUDA. A segunda instala `ffmpeg`, Whisper e o Qwen3-TTS oficial. Ela fica recolhida como formulário porque é infraestrutura, não a parte central do aprendizado.
    """
)

code(
    """
    import gc
    import importlib
    import importlib.util
    import json
    import os
    import random
    import shutil
    import subprocess
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não encontrada. Selecione Ambiente de execução → "
            "Alterar tipo de ambiente de execução → GPU."
        )

    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {GPU_NAME} | memória: {GPU_MEMORY_GB:.1f} GB")
    """
)

code(
    """
    QWEN_REPO = Path("/content/Qwen3-TTS")
    if not QWEN_REPO.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/QwenLM/Qwen3-TTS.git", str(QWEN_REPO)],
            check=True,
        )

    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "ffmpeg", "sox", "libsox-fmt-all"],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--upgrade", str(QWEN_REPO),
            "openai-whisper", "pydub", "soundfile", "accelerate",
            "tensorboard", "huggingface_hub", "safetensors",
        ],
        check=True,
    )
    QWEN_COMMIT = subprocess.check_output(
        ["git", "-C", str(QWEN_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    importlib.invalidate_caches()
    if importlib.util.find_spec("qwen_tts") is None:
        raise RuntimeError(
            "qwen_tts não ficou disponível no kernel. Reinicie a sessão e execute "
            "as células desde o início."
        )
    from qwen_tts import Qwen3TTSModel as _Qwen3TTSModel

    print("Qwen3-TTS preparado no commit:", QWEN_COMMIT)
    print("Import qwen_tts validado no kernel atual.")
    del _Qwen3TTSModel
    """,
    compact=True,
)

md(
    """
    ### 1.1 Configuração e upload da gravação

    O painel **Arquivos** do Colab mostra o disco remoto `/content`, não uma pasta do Mac. Ao executar esta célula, escolha `voz_ptbr.wav` no seletor. Tudo será armazenado temporariamente em `/content/FineTunning-storage/<run_id>` e baixado como um pacote ao final.

    As variáveis abaixo concentram as decisões do experimento. Para o primeiro teste, altere apenas `SPEAKER_NAME` se desejar.
    """
)

code(
    """
    from google.colab import files

    SPEAKER_NAME = "felipe"
    RAW_AUDIO_FILENAME = "voz_ptbr.wav"
    LANGUAGE = "Portuguese"
    WHISPER_LANGUAGE = "pt"
    WHISPER_MODEL = "medium"
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-6
    REFERENCE_INDEX = 0
    RANDOM_SEED = 42

    gpu_upper = GPU_NAME.upper()
    if "T4" in gpu_upper or GPU_MEMORY_GB < 20:
        MODEL_SIZE, MIXED_PRECISION, BATCH_SIZE = "0.6B", "fp16", 1
    elif "L4" in gpu_upper or GPU_MEMORY_GB < 40:
        MODEL_SIZE, MIXED_PRECISION, BATCH_SIZE = "1.7B", "bf16", 2
    else:
        MODEL_SIZE, MIXED_PRECISION, BATCH_SIZE = "1.7B", "bf16", 4

    MODEL_ID = f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base"
    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{SPEAKER_NAME}"
    RUN_DIR = Path("/content/FineTunning-storage") / RUN_ID
    for name in ["data/raw", "data/chunks", "samples", "checkpoints"]:
        (RUN_DIR / name).mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    raw_path = RUN_DIR / "data" / "raw" / RAW_AUDIO_FILENAME
    sidebar_path = Path("/content") / RAW_AUDIO_FILENAME
    if sidebar_path.exists():
        shutil.copy2(sidebar_path, raw_path)
    else:
        print(f"Escolha {RAW_AUDIO_FILENAME} no seu Mac.")
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise RuntimeError("Envie exatamente um arquivo WAV.")
        uploaded_name, uploaded_bytes = next(iter(uploaded.items()))
        if not uploaded_name.lower().endswith(".wav"):
            raise ValueError("O arquivo precisa ter extensão .wav.")
        raw_path.write_bytes(uploaded_bytes)

    config = {
        "run_id": RUN_ID, "speaker_name": SPEAKER_NAME, "model_id": MODEL_ID,
        "gpu": GPU_NAME, "mixed_precision": MIXED_PRECISION,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS, "qwen_commit": QWEN_COMMIT,
    }
    (RUN_DIR / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    """
)

md(
    """
    ## 2. Parte A — Preparação dos dados (`02_data_prep.ipynb`)

    ### 2.1 Inspecionar, normalizar e segmentar — original §§2–3

    Primeiro conferimos duração, canais e sample rate. Em seguida, convertemos para mono/24 kHz e dividimos a gravação nas pausas. O treino trabalha melhor com exemplos curtos; manteremos somente trechos entre 2 e 15 segundos.

    Escute os primeiros vinte segundos. Se houver distorção, eco forte ou ruído constante, regravar costuma ser melhor do que tentar “consertar” o áudio com filtros agressivos.
    """
)

code(
    """
    import numpy as np
    import soundfile as sf
    from IPython.display import Audio, display
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    info = sf.info(str(raw_path))
    print(info)
    if not 4 <= info.duration / 60 <= 12:
        print(f"ATENÇÃO: duração de {info.duration / 60:.1f} min; esperávamos 5–10 min.")

    normalized_path = RUN_DIR / "data" / "voz_24k_mono.wav"
    recording = AudioSegment.from_file(str(raw_path)).set_frame_rate(24_000).set_channels(1)
    recording.export(normalized_path, format="wav")
    waveform, sample_rate = sf.read(normalized_path)
    clipping = float(np.mean(np.abs(waveform) >= 0.99))
    print(f"Normalizado: {sample_rate} Hz | clipping: {clipping:.4%}")
    display(Audio(waveform[: sample_rate * 20], rate=sample_rate))

    chunks_dir = RUN_DIR / "data" / "chunks"
    raw_chunks = split_on_silence(
        recording,
        min_silence_len=550,
        silence_thresh=recording.dBFS - 18,
        keep_silence=120,
    )
    utterances = []
    for index, chunk in enumerate(raw_chunks):
        duration = len(chunk) / 1000
        if 2 <= duration <= 15:
            path = chunks_dir / f"utt_{index:04d}.wav"
            chunk.export(path, format="wav")
            utterances.append({"audio": path, "duration": duration})

    print(f"Trechos: {len(utterances)} | duração útil: {sum(x['duration'] for x in utterances) / 60:.1f} min")
    if len(utterances) < 20:
        raise RuntimeError("Poucos trechos. Verifique as pausas e a qualidade da gravação.")
    """
)

md(
    """
    ### 2.2 Transcrever e revisar — original §§4–5

    Whisper produz o texto associado a cada trecho. Esse texto é o rótulo supervisionado: uma transcrição errada ensina uma relação errada entre escrita e som. Depois da transcrição, ouviremos dez exemplos aleatórios.

    Para corrigir um item, execute `utterances[ÍNDICE]["text"] = "texto correto"` antes de continuar.
    """
)

code(
    """
    import whisper

    whisper_model = whisper.load_model(WHISPER_MODEL, device="cuda")
    transcribed = []
    for index, item in enumerate(utterances, start=1):
        result = whisper_model.transcribe(
            str(item["audio"]), language=WHISPER_LANGUAGE,
            beam_size=5, temperature=0.0, condition_on_previous_text=False,
        )
        text = result["text"].strip()
        if len(text) >= 5:
            transcribed.append({**item, "text": text})
        if index % 10 == 0 or index == len(utterances):
            print(f"Transcritos: {index}/{len(utterances)}")

    utterances = transcribed
    del whisper_model
    torch.cuda.empty_cache()
    print("Trechos mantidos:", len(utterances))
    """
)

code(
    """
    for index in random.sample(range(len(utterances)), min(10, len(utterances))):
        item = utterances[index]
        print(f"Índice {index} | {item['duration']:.1f}s | {item['text']}")
        display(Audio(filename=str(item["audio"])))

    print("Correção: utterances[ÍNDICE]['text'] = 'Transcrição correta.'")
    """
)

md(
    """
    ### 2.3 Criar JSONL e códigos acústicos — original §§6–7

    Cada linha de `train_raw.jsonl` contém `audio`, `text` e `ref_audio`. Escolhemos um trecho limpo como referência e usamos o mesmo arquivo em todo o dataset, como recomenda o Qwen. Depois, o tokenizer de 12 Hz converte cada WAV em `audio_codes`.

    Ouça a referência abaixo. Se ela contiver hesitação ou ruído, mude `REFERENCE_INDEX` na configuração e reexecute esta célula.
    """
)

code(
    """
    if not 0 <= REFERENCE_INDEX < len(utterances):
        raise IndexError("REFERENCE_INDEX não existe.")

    reference = utterances[REFERENCE_INDEX]
    reference_path = RUN_DIR / "data" / "reference.wav"
    shutil.copy2(reference["audio"], reference_path)
    reference_text = reference["text"]
    (RUN_DIR / "data" / "reference.txt").write_text(reference_text, encoding="utf-8")
    print("Referência:", reference_text)
    display(Audio(filename=str(reference_path)))

    train_raw = RUN_DIR / "data" / "train_raw.jsonl"
    with train_raw.open("w", encoding="utf-8") as handle:
        for item in utterances:
            row = {
                "audio": str(item["audio"].resolve()),
                "text": item["text"],
                "ref_audio": str(reference_path.resolve()),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\\n")
    print("Linhas no manifesto:", len(utterances))
    """
)

code(
    """
    train_codes = RUN_DIR / "data" / "train_with_codes.jsonl"
    result = subprocess.run(
        [
            sys.executable, str(QWEN_REPO / "finetuning" / "prepare_data.py"),
            "--device", "cuda:0",
            "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "--input_jsonl", str(train_raw),
            "--output_jsonl", str(train_codes),
        ],
        text=True,
    )
    if result.returncode != 0 or not train_codes.exists():
        raise RuntimeError("Falha ao extrair audio_codes; consulte a saída acima.")
    print("Dataset pronto:", train_codes)
    """
)

md(
    """
    ## 3. Parte B — Clonagem zero-shot (`01_voice_cloning.ipynb`)

    ### 3.1 Carregar modelo, criar voice prompt e testar — original §§2–6

    Esta é nossa linha de base antes do treino. O modelo recebe o WAV de referência e sua transcrição, cria um `voice_clone_prompt` e gera três frases inéditas. O original também demonstra presets, x-vector e histórias longas; eles foram omitidos porque não são necessários para avaliar o fine-tuning.

    O primeiro carregamento baixa vários gigabytes. A T4 usa 0.6B; GPUs maiores usam 1.7B.
    """
)

code(
    """
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    TEST_SENTENCES = [
        "Este é um teste de voz com uma frase que não apareceu durante o treinamento.",
        "Você confirmou a reunião para quinta-feira, às nove horas e trinta minutos?",
        "Que resultado incrível! Finalmente conseguimos concluir o experimento com sucesso.",
    ]
    MODEL_PATH = Path(snapshot_download(MODEL_ID, cache_dir="/content/huggingface-cache"))
    TORCH_DTYPE = torch.float16 if MIXED_PRECISION == "fp16" else torch.bfloat16
    base_model = Qwen3TTSModel.from_pretrained(
        str(MODEL_PATH), device_map="cuda:0",
        dtype=TORCH_DTYPE, attn_implementation="sdpa",
    )
    voice_prompt = base_model.create_voice_clone_prompt(
        ref_audio=str(reference_path), ref_text=reference_text
    )
    """
)

code(
    """
    for index, sentence in enumerate(TEST_SENTENCES, start=1):
        wavs, sr = base_model.generate_voice_clone(
            text=sentence, language=LANGUAGE, voice_clone_prompt=voice_prompt
        )
        output = RUN_DIR / "samples" / f"zero_shot_{index}.wav"
        sf.write(output, wavs[0], sr)
        print(sentence)
        display(Audio(wavs[0], rate=sr))

    del base_model, voice_prompt
    gc.collect()
    torch.cuda.empty_cache()
    """
)

md(
    """
    ## 4. Parte C — Fine-tuning e teste (`03_finetune.ipynb`)

    ### 4.1 Preparar e treinar — original §2

    O script oficial assume BF16 e FlashAttention. Para também funcionar em T4, fazemos somente duas alterações: escolhemos FP16/BF16 pela GPU e usamos `sdpa`. O modelo-base já foi baixado localmente, evitando o problema do script ao copiar um identificador remoto.

    Esta célula de compatibilidade fica recolhida. Se o Qwen alterar o script oficial, ela interrompe com uma mensagem em vez de aplicar um patch incorreto.
    """
)

code(
    """
    finetuning_dir = RUN_DIR / "qwen_finetuning"
    shutil.copytree(QWEN_REPO / "finetuning", finetuning_dir, dirs_exist_ok=True)
    train_script = finetuning_dir / "sft_12hz.py"
    source = train_script.read_text(encoding="utf-8")

    patches = {
        'accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")':
            'precision = os.environ.get("QWEN_MIXED_PRECISION", "bf16")\\n    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision=precision, log_with="tensorboard")',
        'torch_dtype=torch.bfloat16,\\n        attn_implementation="flash_attention_2",':
            'torch_dtype=torch.bfloat16 if precision == "bf16" else torch.float16,\\n        attn_implementation="sdpa",',
    }
    for old, new in patches.items():
        if old not in source:
            raise RuntimeError("O script oficial mudou; revise o patch de compatibilidade.")
        source = source.replace(old, new, 1)
    train_script.write_text(source, encoding="utf-8")
    print("Script de treino preparado:", train_script)
    """,
    compact=True,
)

code(
    """
    train_log = RUN_DIR / "train.log"
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["QWEN_MIXED_PRECISION"] = MIXED_PRECISION
    command = [
        sys.executable, str(train_script),
        "--init_model_path", str(MODEL_PATH),
        "--output_model_path", str(RUN_DIR / "checkpoints"),
        "--train_jsonl", str(train_codes),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LEARNING_RATE),
        "--num_epochs", str(NUM_EPOCHS),
        "--speaker_name", SPEAKER_NAME,
    ]

    print("Iniciando a etapa mais demorada do notebook.")
    with train_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command, cwd=finetuning_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("Treinamento falhou. Consulte train.log.")

    checkpoints = sorted(
        (RUN_DIR / "checkpoints").glob("checkpoint-epoch-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    for stale in checkpoints[:-2]:
        shutil.rmtree(stale)
    checkpoints = checkpoints[-2:]
    print("Checkpoints mantidos:", [path.name for path in checkpoints])
    """
)

md(
    """
    ### 4.2 Carregar o checkpoint e comparar — original §3

    O checkpoint treinado funciona como `CustomVoice`: informamos o nome do locutor em vez de um áudio de referência. Geraremos as mesmas frases do zero-shot. Compare identidade, inteligibilidade, naturalidade e estabilidade; perda menor no treino não garante som melhor.

    O original ainda sintetiza uma história completa e publica no Hugging Face. Essas etapas ficam fora deste primeiro experimento para manter o foco na comparação.
    """
)

code(
    """
    if not checkpoints:
        raise FileNotFoundError("Nenhum checkpoint foi criado.")
    selected_checkpoint = checkpoints[-1]
    fine_tuned_model = Qwen3TTSModel.from_pretrained(
        str(selected_checkpoint), device_map="cuda:0",
        dtype=TORCH_DTYPE, attn_implementation="sdpa",
    )

    for index, sentence in enumerate(TEST_SENTENCES, start=1):
        wavs, sr = fine_tuned_model.generate_custom_voice(
            text=sentence, language=LANGUAGE, speaker=SPEAKER_NAME
        )
        output = RUN_DIR / "samples" / f"fine_tuned_{index}.wav"
        sf.write(output, wavs[0], sr)
        print(f"Frase {index} — zero-shot")
        display(Audio(filename=str(RUN_DIR / "samples" / f"zero_shot_{index}.wav")))
        print(f"Frase {index} — fine-tuned")
        display(Audio(wavs[0], rate=sr))

    del fine_tuned_model
    gc.collect()
    torch.cuda.empty_cache()
    """
)

md(
    """
    ## 5. Exportar os resultados para o Mac

    O disco `/content` é temporário. Esta célula empacota configuração, áudio processado, logs, amostras e os dois checkpoints recentes em um `.tar` sem tentar recomprimir os pesos. Aguarde o download terminar antes de fechar a sessão.

    Se o navegador bloquear um arquivo grande, encontre `/content/<run_id>.tar` no painel **Arquivos**, abra seu menu e escolha **Fazer download**.
    """
)

code(
    """
    export_path = Path("/content") / f"{RUN_ID}.tar"
    subprocess.run(
        ["tar", "-cf", str(export_path), "-C", str(RUN_DIR.parent), RUN_DIR.name],
        check=True,
    )
    print(f"Pacote: {export_path} | {export_path.stat().st_size / 1024**3:.2f} GB")
    files.download(str(export_path))
    """
)

md(
    """
    ## Apêndice — retomada de uma execução interrompida

    A retomada não faz parte do primeiro fluxo porque adiciona complexidade e o otimizador oficial não é salvo. Um checkpoint concluído pode ser reutilizado, mas é necessário restaurar o `speaker_encoder` do modelo-base e reiniciar o otimizador. Isso não equivale exatamente a continuar do mesmo passo.

    Para um primeiro corpus de 5–10 minutos, prefira concluir poucas épocas em uma única sessão e baixar o pacote final. Quando o fluxo básico estiver validado, a retomada pode ser adicionada como um notebook avançado separado.
    """
)

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Notebook gerado: {OUTPUT} ({len(cells)} células)")
