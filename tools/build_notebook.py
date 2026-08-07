"""Gera o notebook didático sem depender de nbformat."""

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


def code(source: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n",
        }
    )


md(
    """
    <a href="https://colab.research.google.com/github/FelipeFMMobile/finetunningaudio/blob/main/notebooks/01_qwen3_tts_clone_finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir no Colab"/></a>

    # Clonagem e fine-tuning de voz com Qwen3-TTS

    Este notebook é uma aula prática em português brasileiro. Ao final, você terá construído um corpus a partir de uma gravação própria, criado uma linha de base com clonagem *zero-shot*, treinado um modelo single-speaker e comparado os resultados.

    > **Uso responsável:** prossiga somente com sua própria voz ou com consentimento explícito. Voz, transcrições e pesos podem ser dados pessoais. Eles serão mantidos no Google Drive e não devem ser enviados ao Git.

    **Como estudar:** execute as células na ordem. Leia primeiro o texto que antecede cada código e confira o resultado esperado antes de avançar. Células de instalação, transcrição, tokenização e treino podem levar vários minutos.
    """
)

md(
    """
    ## 1. Visão geral do projeto

    **Objetivo de aprendizagem:** distinguir síntese de voz, clonagem *zero-shot* e *fine-tuning*.

    Um sistema TTS (*text-to-speech*) transforma texto em fala. Na clonagem *zero-shot*, o modelo usa poucos segundos de áudio como referência durante a geração, sem alterar seus pesos. No *fine-tuning*, ajustamos os pesos do modelo com vários pares de áudio e transcrição para criar uma identidade de locutor persistente.

    O Qwen3-TTS representa fala com um tokenizer acústico de 12 Hz. Ele converte o áudio em códigos discretos, de forma semelhante à conversão de palavras em tokens feita por um LLM. O modelo aprende a prever esses códigos condicionado pelo texto e por uma representação do locutor.

    **Relação com o curso:** seguimos a mesma sequência `voice cloning → data prep → fine-tuning`, mas reunida em um notebook reexecutável, com persistência no Drive e adaptação à GPU do Colab.

    **Resultado esperado:** compreender que a clonagem zero-shot será nossa linha de base e que o treino somente começa depois da validação do corpus.
    """
)

md(
    """
    ## 2. Ambiente Google Colab

    **Objetivo de aprendizagem:** identificar a GPU e entender como ela influencia modelo, precisão e batch.

    O navegador roda no seu Mac, mas as células são executadas em uma máquina temporária do Google. CUDA é a plataforma usada pelo PyTorch para acessar GPUs NVIDIA. Uma T4 tem menos memória e não oferece o mesmo suporte eficiente a BF16 de uma L4 ou A100; por isso usaremos FP16 e o modelo 0.6B nela.

    Antes de executar, selecione **Ambiente de execução → Alterar tipo de ambiente de execução → GPU**. A célula abaixo falha cedo e com uma mensagem clara se CUDA não estiver disponível.

    **Resultado esperado:** nome da GPU, memória aproximada e versão CUDA serão exibidos.
    """
)

code(
    """
    import json
    import os
    import platform
    import random
    import shutil
    import subprocess
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não foi encontrada. No Colab, selecione Ambiente de execução → "
            "Alterar tipo de ambiente de execução → GPU e reinicie a sessão."
        )

    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {GPU_NAME}")
    print(f"Memória: {GPU_MEMORY_GB:.1f} GB")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    """
)

md(
    """
    ### 2.1 Instalação reproduzível

    A sessão do Colab é descartável. Instalamos `ffmpeg` para conversão de áudio, o Qwen3-TTS oficial, Whisper para transcrição e bibliotecas auxiliares. O repositório oficial é clonado uma única vez na sessão e seu commit será registrado junto ao experimento.

    Usamos `sdpa`, a implementação de atenção do PyTorch, para evitar a compilação demorada do FlashAttention. Isso favorece compatibilidade entre T4, L4 e A100.

    **Alerta:** reinicie o ambiente somente se o `pip` solicitar. Uma nova sessão exige executar esta célula novamente.
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
    subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            str(QWEN_REPO),
            "openai-whisper",
            "pydub",
            "soundfile",
            "accelerate",
            "tensorboard",
            "huggingface_hub",
            "safetensors",
        ],
        check=True,
    )
    QWEN_COMMIT = subprocess.check_output(
        ["git", "-C", str(QWEN_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    print("Qwen3-TTS preparado no commit:", QWEN_COMMIT)
    """
)

md(
    """
    ## 3. Google Drive e persistência

    **Objetivo de aprendizagem:** separar armazenamento temporário e persistente.

    `/content` é rápido, porém desaparece quando a sessão termina. O Drive é persistente, mas é lento para milhares de arquivos pequenos. Portanto, copiamos a gravação para `/content`, treinamos ali e sincronizamos resultados importantes para `MyDrive/FineTunning`.

    A montagem abaixo pedirá autorização da sua conta Google. Nenhum token será gravado no notebook.

    **Resultado esperado:** as pastas `data`, `runs` e `exports` existirão no Drive.
    """
)

code(
    """
    from google.colab import drive

    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive/FineTunning")
    for relative in [
        "data/raw", "data/reference", "data/processed", "runs", "exports"
    ]:
        (DRIVE_ROOT / relative).mkdir(parents=True, exist_ok=True)

    free_drive_gb = shutil.disk_usage(DRIVE_ROOT).free / 1024**3
    print(f"Drive montado em: {DRIVE_ROOT}")
    print(f"Espaço livre reportado: {free_drive_gb:.1f} GB")
    if free_drive_gb < 8:
        print("ATENÇÃO: menos de 8 GB livres; o modelo 1.7B pode não caber com dois checkpoints.")
    """
)

md(
    """
    ## 4. Configuração central do experimento

    **Objetivo de aprendizagem:** relacionar hiperparâmetros com memória, velocidade e estabilidade.

    O modo `auto` seleciona 0.6B/FP16/batch 1 em T4, 1.7B/BF16/batch 2 em L4 e batch 4 em A100. O *learning rate* controla o tamanho de cada atualização; um valor alto pode apagar capacidades do modelo, enquanto um valor muito baixo pode produzir pouca adaptação.

    `RUN_ID_OVERRIDE` permite reabrir uma execução anterior. Deixe vazio na primeira vez. Para continuar após uma interrupção, copie o `run_id` salvo no Drive para essa variável e informe `RESUME_CHECKPOINT` mais adiante.

    **Resultado esperado:** um `config.json` imutável será salvo na pasta da execução.
    """
)

code(
    """
    # Configurações que você pode editar.
    SPEAKER_NAME = "felipe"
    RAW_AUDIO_FILENAME = "voz_ptbr.wav"
    LANGUAGE = "Portuguese"
    WHISPER_LANGUAGE = "pt"
    WHISPER_MODEL = "medium"
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-6
    KEEP_CHECKPOINTS = 2
    REFERENCE_INDEX = 0
    RANDOM_SEED = 42
    RUN_ID_OVERRIDE = ""  # Exemplo: 20260806T210300Z_felipe

    gpu_upper = GPU_NAME.upper()
    if "T4" in gpu_upper or GPU_MEMORY_GB < 20:
        MODEL_SIZE, MIXED_PRECISION, BATCH_SIZE = "0.6B", "fp16", 1
    elif "L4" in gpu_upper or GPU_MEMORY_GB < 40:
        MODEL_SIZE, MIXED_PRECISION, BATCH_SIZE = "1.7B", "bf16", 2
    else:
        MODEL_SIZE, MIXED_PRECISION, BATCH_SIZE = "1.7B", "bf16", 4

    MODEL_ID = f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base"
    RUN_ID = RUN_ID_OVERRIDE or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{SPEAKER_NAME}"
    )
    WORK_ROOT = Path("/content/FineTunning-work") / RUN_ID
    DRIVE_RUN = DRIVE_ROOT / "runs" / RUN_ID
    for path in [WORK_ROOT / "data", WORK_ROOT / "checkpoints", WORK_ROOT / "samples",
                 DRIVE_RUN / "checkpoints", DRIVE_RUN / "samples"]:
        path.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    config = {
        "run_id": RUN_ID,
        "speaker_name": SPEAKER_NAME,
        "language": LANGUAGE,
        "model_id": MODEL_ID,
        "gpu": GPU_NAME,
        "gpu_memory_gb": round(GPU_MEMORY_GB, 2),
        "mixed_precision": MIXED_PRECISION,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "qwen_commit": QWEN_COMMIT,
        "random_seed": RANDOM_SEED,
    }
    (DRIVE_RUN / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    """
)

md(
    """
    ## 5. Gravação do corpus

    **Objetivo de aprendizagem:** reconhecer como a qualidade da entrada limita a qualidade do modelo.

    Grave o roteiro `scripts/roteiro_gravacao_ptbr.txt` em um local silencioso. Mantenha distância, ganho e posição do microfone constantes. Faça uma pausa curta entre parágrafos; ela ajudará a segmentação. Se errar, espere dois segundos e recomece o parágrafo.

    Envie o WAV para `MyDrive/FineTunning/data/raw/voz_ptbr.wav`. A célula copia o original para o armazenamento rápido sem modificá-lo.

    **Erros comuns:** gravação com música, eco, cancelamento agressivo de ruído, clipping ou mudança de microfone no meio do corpus.
    """
)

code(
    """
    RAW_DRIVE_PATH = DRIVE_ROOT / "data" / "raw" / RAW_AUDIO_FILENAME
    if not RAW_DRIVE_PATH.exists():
        raise FileNotFoundError(
            f"Envie sua gravação para {RAW_DRIVE_PATH} e execute a célula novamente."
        )
    RAW_LOCAL_PATH = WORK_ROOT / "data" / RAW_AUDIO_FILENAME
    shutil.copy2(RAW_DRIVE_PATH, RAW_LOCAL_PATH)
    print("Cópia de trabalho criada:", RAW_LOCAL_PATH)
    """
)

md(
    """
    ## 6. Inspeção e normalização

    **Objetivo de aprendizagem:** interpretar sample rate, canais, duração e clipping.

    O *sample rate* informa quantas amostras representam um segundo. O Qwen3-TTS espera áudio mono a 24 kHz na preparação. Clipping ocorre quando a amplitude encosta repetidamente no limite e causa distorção irreversível.

    A normalização abaixo converte formato e canais, mas não aplica efeitos nem compressão dinâmica. O original permanece intacto no Drive.

    **Resultado esperado:** áudio entre 5 e 10 minutos, mono, 24 kHz e com baixa proporção de amostras próximas do limite.
    """
)

code(
    """
    import numpy as np
    import soundfile as sf
    from IPython.display import Audio, display
    from pydub import AudioSegment

    original_info = sf.info(str(RAW_LOCAL_PATH))
    print(original_info)
    duration_min = original_info.duration / 60
    if not 4 <= duration_min <= 12:
        print(f"ATENÇÃO: duração de {duration_min:.1f} min fora da faixa esperada de 5–10 min.")

    normalized_path = WORK_ROOT / "data" / "voz_normalizada_24k_mono.wav"
    segment = AudioSegment.from_file(str(RAW_LOCAL_PATH)).set_frame_rate(24_000).set_channels(1)
    segment.export(str(normalized_path), format="wav")
    waveform, sample_rate = sf.read(normalized_path)
    clipping_ratio = float(np.mean(np.abs(waveform) >= 0.99))
    print(f"Normalizado: {sample_rate} Hz, mono, {len(waveform) / sample_rate / 60:.1f} min")
    print(f"Amostras próximas de clipping: {clipping_ratio:.4%}")
    if clipping_ratio > 0.001:
        print("ATENÇÃO: há indícios de clipping. Considere regravar com ganho menor.")
    display(Audio(waveform[: sample_rate * 20], rate=sample_rate))
    """
)

md(
    """
    ## 7. Segmentação por silêncio

    **Objetivo de aprendizagem:** entender por que exemplos curtos facilitam o alinhamento entre texto e áudio.

    Uma gravação longa contém muitas frases. O treinamento funciona melhor quando cada exemplo representa uma unidade curta e coerente. Detectamos pausas, preservamos 120 ms nas bordas e aceitamos trechos entre 2 e 15 segundos.

    O limiar de silêncio é calculado em relação ao volume médio da própria gravação. Se quase nenhum trecho for produzido, aumente `MIN_SILENCE_MS`; se houver cortes no meio de palavras, reduza a sensibilidade usando um valor mais baixo para `SILENCE_THRESHOLD_OFFSET_DB`.

    **Resultado esperado:** dezenas de WAVs curtos em uma pasta exclusiva desta execução.
    """
)

code(
    """
    from pydub.silence import split_on_silence

    MIN_DURATION_S = 2.0
    MAX_DURATION_S = 15.0
    MIN_SILENCE_MS = 550
    KEEP_SILENCE_MS = 120
    SILENCE_THRESHOLD_OFFSET_DB = 18

    chunks_dir = WORK_ROOT / "data" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    silence_threshold = segment.dBFS - SILENCE_THRESHOLD_OFFSET_DB
    raw_chunks = split_on_silence(
        segment,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=silence_threshold,
        keep_silence=KEEP_SILENCE_MS,
    )
    utterances = []
    for index, chunk in enumerate(raw_chunks):
        duration = len(chunk) / 1000
        if MIN_DURATION_S <= duration <= MAX_DURATION_S:
            path = chunks_dir / f"utt_{index:04d}.wav"
            chunk.export(path, format="wav")
            utterances.append({"audio": path, "duration": duration})

    total_minutes = sum(item["duration"] for item in utterances) / 60
    print(f"Trechos aceitos: {len(utterances)} | duração útil: {total_minutes:.1f} min")
    if len(utterances) < 20:
        raise RuntimeError("Poucos trechos foram produzidos. Revise pausas e parâmetros de silêncio.")
    """
)

md(
    """
    ## 8. Transcrição automática e revisão

    **Objetivo de aprendizagem:** compreender que transcrição é o rótulo supervisionado do áudio.

    Whisper converte cada trecho em texto. Um rótulo incorreto ensina ao modelo uma relação errada entre escrita e som; por isso a revisão auditiva é parte do pipeline, não uma etapa decorativa. Usamos português, decodificação determinística e não carregamos contexto entre trechos.

    A célula pode levar vários minutos. Em seguida, serão exibidas amostras aleatórias. Para corrigir um texto, edite diretamente `utterances[i]["text"]` em uma nova célula antes de continuar.

    **Resultado esperado:** todos os exemplos mantidos terão texto não vazio e será possível ouvir uma amostra.
    """
)

code(
    """
    import whisper

    whisper_model = whisper.load_model(WHISPER_MODEL, device="cuda")
    transcribed = []
    for index, item in enumerate(utterances, start=1):
        result = whisper_model.transcribe(
            str(item["audio"]),
            language=WHISPER_LANGUAGE,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = result["text"].strip()
        if len(text) >= 5:
            transcribed.append({**item, "text": text})
        if index % 10 == 0 or index == len(utterances):
            print(f"Transcritos: {index}/{len(utterances)}")
    utterances = transcribed
    print("Trechos mantidos:", len(utterances))
    del whisper_model
    torch.cuda.empty_cache()
    """
)

code(
    """
    review_count = min(10, len(utterances))
    for index in random.sample(range(len(utterances)), review_count):
        item = utterances[index]
        print(f"Índice {index} | {item['duration']:.1f}s | {item['text']}")
        display(Audio(filename=str(item["audio"])))

    print("Para corrigir: utterances[ÍNDICE]['text'] = 'Transcrição corrigida.'")
    """
)

md(
    """
    ## 9. Manifestos JSONL e códigos acústicos

    **Objetivo de aprendizagem:** conhecer a interface de dados usada pelo fine-tuning oficial.

    JSONL contém um objeto JSON por linha. `audio` aponta para o trecho, `text` contém sua transcrição e `ref_audio` aponta para uma única referência do locutor. A recomendação oficial é manter a mesma referência em todo o dataset para melhorar consistência.

    Escolhemos um trecho limpo como referência. Ouça o índice configurado e altere `REFERENCE_INDEX` se ele tiver ruído, hesitação ou transcrição incorreta. O tokenizer Qwen acrescentará `audio_codes`, a representação discreta utilizada como alvo de treinamento.

    **Resultado esperado:** `train_raw.jsonl`, `train_with_codes.jsonl` e um arquivo compactado no Drive.
    """
)

code(
    """
    if not 0 <= REFERENCE_INDEX < len(utterances):
        raise IndexError("REFERENCE_INDEX não existe na lista de trechos.")
    reference = utterances[REFERENCE_INDEX]
    reference_path = WORK_ROOT / "data" / "reference.wav"
    shutil.copy2(reference["audio"], reference_path)
    reference_text = reference["text"]
    (DRIVE_ROOT / "data" / "reference" / f"{RUN_ID}.txt").write_text(reference_text, encoding="utf-8")
    shutil.copy2(reference_path, DRIVE_ROOT / "data" / "reference" / f"{RUN_ID}.wav")
    print("Referência:", reference_text)
    display(Audio(filename=str(reference_path)))

    train_raw = WORK_ROOT / "data" / "train_raw.jsonl"
    with train_raw.open("w", encoding="utf-8") as handle:
        for item in utterances:
            row = {
                "audio": str(item["audio"].resolve()),
                "text": item["text"],
                "ref_audio": str(reference_path.resolve()),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\\n")
    print("Manifesto bruto:", train_raw, "| linhas:", len(utterances))
    """
)

code(
    """
    train_codes = WORK_ROOT / "data" / "train_with_codes.jsonl"
    prepare_script = QWEN_REPO / "finetuning" / "prepare_data.py"
    result = subprocess.run(
        [
            sys.executable, str(prepare_script),
            "--device", "cuda:0",
            "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "--input_jsonl", str(train_raw),
            "--output_jsonl", str(train_codes),
        ],
        text=True,
    )
    if result.returncode != 0 or not train_codes.exists():
        raise RuntimeError("A extração dos códigos acústicos falhou; consulte a saída acima.")

    archive_base = WORK_ROOT / f"dataset_{RUN_ID}"
    archive_path = Path(shutil.make_archive(str(archive_base), "gztar", root_dir=WORK_ROOT / "data"))
    archive_drive = DRIVE_ROOT / "data" / "processed" / archive_path.name
    shutil.copy2(archive_path, archive_drive)
    print("Dataset tokenizado:", train_codes)
    print("Backup compactado:", archive_drive)
    """
)

md(
    """
    ## 10. Clonagem zero-shot: nossa linha de base

    **Objetivo de aprendizagem:** estabelecer uma comparação antes de alterar o modelo.

    A clonagem zero-shot usa `reference.wav` e sua transcrição para gerar frases inéditas. Salvamos três amostras: uma declarativa, uma pergunta com números e uma frase expressiva. As mesmas frases serão usadas após o treinamento.

    **Alerta:** o primeiro carregamento baixa vários gigabytes. Não confunda tempo de download com tempo de inferência.

    **Resultado esperado:** três WAVs `zero_shot_*.wav` sincronizados na pasta `samples` do Drive.
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
        str(MODEL_PATH), device_map="cuda:0", dtype=TORCH_DTYPE, attn_implementation="sdpa"
    )
    for index, sentence in enumerate(TEST_SENTENCES, start=1):
        wavs, sr = base_model.generate_voice_clone(
            text=sentence,
            language=LANGUAGE,
            ref_audio=str(reference_path),
            ref_text=reference_text,
        )
        output = WORK_ROOT / "samples" / f"zero_shot_{index}.wav"
        sf.write(output, wavs[0], sr)
        shutil.copy2(output, DRIVE_RUN / "samples" / output.name)
        print(sentence)
        display(Audio(wavs[0], rate=sr))
    del base_model
    torch.cuda.empty_cache()
    """
)

md(
    """
    ## 11. Fine-tuning single-speaker

    **Objetivo de aprendizagem:** executar o treinamento e interpretar seus principais controles.

    O script oficial combina a perda do primeiro codebook com a perda dos codebooks auxiliares. `batch_size` controla quantos exemplos são processados juntos; a acumulação de gradiente simula um batch maior. Cada época percorre o corpus inteiro.

    Criamos uma cópia adaptada do script oficial somente dentro de `/content`: BF16/FP16 é escolhido pela GPU, `sdpa` substitui FlashAttention e cada checkpoint completo é copiado imediatamente para o Drive. A revisão oficial usada está registrada em `config.json`.

    **Retomada:** se uma sessão cair, defina `RUN_ID_OVERRIDE` na seção 4 e `RESUME_CHECKPOINT` abaixo. A rotina recompõe o `speaker_encoder` que o checkpoint final omite por projeto.

    **Resultado esperado:** logs de perda e pelo menos um `checkpoint-epoch-N` no Drive.
    """
)

code(
    """
    # Deixe vazio em um treino novo. Exemplo para retomada: "checkpoint-epoch-1"
    RESUME_CHECKPOINT = ""

    ft_dir = WORK_ROOT / "finetuning"
    shutil.copytree(QWEN_REPO / "finetuning", ft_dir, dirs_exist_ok=True)
    adaptive_script = ft_dir / "sft_12hz.py"
    source = adaptive_script.read_text(encoding="utf-8")

    replacements = {
        'accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")':
            'precision = os.environ.get("QWEN_MIXED_PRECISION", "bf16")\\n    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision=precision, log_with="tensorboard")',
        'torch_dtype=torch.bfloat16,\\n        attn_implementation="flash_attention_2",':
            'torch_dtype=torch.bfloat16 if precision == "bf16" else torch.float16,\\n        attn_implementation="sdpa",',
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError("O script oficial mudou e o patch adaptativo precisa ser revisado.")
        source = source.replace(old, new, 1)

    save_marker = "            save_file(state_dict, save_path)"
    sync_block = "\\n".join([
        "            save_file(state_dict, save_path)",
        "            sync_root = os.environ.get('QWEN_SYNC_DIR')",
        "            if sync_root:",
        "                os.makedirs(sync_root, exist_ok=True)",
        "                sync_target = os.path.join(sync_root, os.path.basename(output_dir))",
        "                shutil.copytree(output_dir, sync_target, dirs_exist_ok=True)",
        "                keep = int(os.environ.get('QWEN_KEEP_CHECKPOINTS', '2'))",
        "                completed = sorted(",
        "                    [name for name in os.listdir(sync_root) if name.startswith('checkpoint-epoch-')],",
        "                    key=lambda name: int(name.rsplit('-', 1)[-1]),",
        "                )",
        "                for stale in completed[:-keep]:",
        "                    shutil.rmtree(os.path.join(sync_root, stale))",
        "                metrics_path = os.environ.get('QWEN_METRICS_PATH')",
        "                if metrics_path and os.path.exists(metrics_path):",
        "                    shutil.copy2(metrics_path, os.path.join(os.path.dirname(sync_root), 'metrics.jsonl'))",
    ])
    if save_marker not in source:
        raise RuntimeError("Ponto de sincronização não encontrado no script oficial.")
    source = source.replace(save_marker, sync_block, 1)

    print_marker = '                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")'
    metrics_block = print_marker + "\\n" + "\\n".join([
        "                metrics_path = os.environ.get('QWEN_METRICS_PATH')",
        "                if accelerator.is_main_process and metrics_path:",
        "                    with open(metrics_path, 'a', encoding='utf-8') as metrics_file:",
        "                        metrics_file.write(json.dumps({'epoch': epoch, 'step': step, 'loss': loss.item()}) + '\\\\n')",
    ])
    if print_marker not in source:
        raise RuntimeError("Ponto de métricas não encontrado no script oficial.")
    source = source.replace(print_marker, metrics_block, 1)
    adaptive_script.write_text(source, encoding="utf-8")
    print("Script adaptativo criado em:", adaptive_script)
    """
)

code(
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    INIT_MODEL_PATH = MODEL_PATH
    if RESUME_CHECKPOINT:
        saved_checkpoint = DRIVE_RUN / "checkpoints" / RESUME_CHECKPOINT
        if not saved_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint de retomada não encontrado: {saved_checkpoint}")
        resume_dir = WORK_ROOT / "resume" / RESUME_CHECKPOINT
        shutil.copytree(saved_checkpoint, resume_dir, dirs_exist_ok=True)
        shutil.copy2(MODEL_PATH / "config.json", resume_dir / "config.json")
        resumed_state = load_file(resume_dir / "model.safetensors", device="cpu")
        with safe_open(MODEL_PATH / "model.safetensors", framework="pt", device="cpu") as base_weights:
            for key in base_weights.keys():
                if key.startswith("speaker_encoder"):
                    resumed_state[key] = base_weights.get_tensor(key)
        save_file(resumed_state, resume_dir / "model.safetensors")
        del resumed_state
        INIT_MODEL_PATH = resume_dir
        print("Checkpoint recomposto para retomada:", INIT_MODEL_PATH)
    else:
        print("Treino novo a partir de:", INIT_MODEL_PATH)
    """
)

code(
    """
    metrics_local = WORK_ROOT / "metrics.jsonl"
    train_log = WORK_ROOT / "train.log"
    env = os.environ.copy()
    env.update({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "QWEN_MIXED_PRECISION": MIXED_PRECISION,
        "QWEN_SYNC_DIR": str(DRIVE_RUN / "checkpoints"),
        "QWEN_KEEP_CHECKPOINTS": str(KEEP_CHECKPOINTS),
        "QWEN_METRICS_PATH": str(metrics_local),
    })
    command = [
        sys.executable, str(adaptive_script),
        "--init_model_path", str(INIT_MODEL_PATH),
        "--output_model_path", str(WORK_ROOT / "checkpoints"),
        "--train_jsonl", str(train_codes),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LEARNING_RATE),
        "--num_epochs", str(NUM_EPOCHS),
        "--speaker_name", SPEAKER_NAME,
    ]
    print("Iniciando treino. Esta é a etapa mais demorada do notebook.")
    with train_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command, cwd=ft_dir, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return_code = process.wait()
    shutil.copy2(train_log, DRIVE_RUN / "train.log")
    if return_code != 0:
        raise RuntimeError("O treinamento terminou com erro. Consulte train.log no Drive.")
    print("Treinamento concluído. Checkpoints:", sorted((DRIVE_RUN / "checkpoints").glob("checkpoint-*")))
    """
)

md(
    """
    ## 12. Avaliação: zero-shot versus modelo treinado

    **Objetivo de aprendizagem:** avaliar identidade, inteligibilidade, naturalidade e estabilidade sem usar frases do corpus.

    O checkpoint mais recente é carregado como `CustomVoice`. Geramos exatamente as mesmas frases da linha de base. Uma melhora de identidade acompanhada de piora na pronúncia pode indicar sobreajuste; áudio acelerado ou instável entre épocas também é um sinal de alerta.

    A avaliação auditiva não é uma métrica perfeita, mas um formulário consistente permite comparar experimentos. Preencha notas de 1 a 5 no CSV criado ao final.

    **Resultado esperado:** pares de áudio comparáveis e uma planilha de avaliação na pasta da execução.
    """
)

code(
    """
    checkpoints = sorted(
        (DRIVE_RUN / "checkpoints").glob("checkpoint-epoch-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError("Nenhum checkpoint completo foi encontrado no Drive.")
    selected_drive_checkpoint = checkpoints[-1]
    selected_local_checkpoint = WORK_ROOT / "evaluation" / selected_drive_checkpoint.name
    shutil.copytree(selected_drive_checkpoint, selected_local_checkpoint, dirs_exist_ok=True)

    fine_tuned_model = Qwen3TTSModel.from_pretrained(
        str(selected_local_checkpoint),
        device_map="cuda:0",
        dtype=TORCH_DTYPE,
        attn_implementation="sdpa",
    )
    for index, sentence in enumerate(TEST_SENTENCES, start=1):
        wavs, sr = fine_tuned_model.generate_custom_voice(
            text=sentence, language=LANGUAGE, speaker=SPEAKER_NAME
        )
        output = WORK_ROOT / "samples" / f"fine_tuned_{index}.wav"
        sf.write(output, wavs[0], sr)
        shutil.copy2(output, DRIVE_RUN / "samples" / output.name)
        print(f"Frase {index} — zero-shot")
        display(Audio(filename=str(DRIVE_RUN / "samples" / f"zero_shot_{index}.wav")))
        print(f"Frase {index} — fine-tuned")
        display(Audio(wavs[0], rate=sr))
    del fine_tuned_model
    torch.cuda.empty_cache()
    """
)

code(
    """
    import csv

    evaluation_csv = DRIVE_RUN / "avaliacao_auditiva.csv"
    with evaluation_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sistema", "frase", "identidade_1a5", "inteligibilidade_1a5",
                        "naturalidade_1a5", "estabilidade_1a5", "observacoes"],
        )
        writer.writeheader()
        for system_name in ["zero_shot", "fine_tuned"]:
            for index in range(1, len(TEST_SENTENCES) + 1):
                writer.writerow({"sistema": system_name, "frase": index})
    print("Formulário de avaliação criado em:", evaluation_csv)
    """
)

md(
    """
    ## 13. Conclusões e próximos experimentos

    **Objetivo de aprendizagem:** transformar observações em uma próxima hipótese testável.

    Compare as notas e o `metrics.jsonl`. Perda menor não garante voz melhor: com apenas 5–10 minutos, muitas épocas podem memorizar o corpus e reduzir a capacidade de pronunciar frases novas. Escute especialmente números, perguntas e palavras ausentes da gravação.

    Próximos experimentos recomendados, um por vez:

    1. corrigir transcrições antes de mudar hiperparâmetros;
    2. ampliar o corpus para 20–30 minutos mantendo o mesmo ambiente;
    3. comparar checkpoints anteriores, não apenas o último;
    4. reduzir épocas ou learning rate se houver fala acelerada, artefatos ou perda de inteligibilidade;
    5. repetir as mesmas frases e o mesmo formulário para preservar comparabilidade.

    **Conclusão produzida:** o Drive contém configuração, revisão do Qwen, métricas, logs, checkpoints e amostras suficientes para reproduzir e analisar esta execução.
    """
)

md(
    """
    ## 14. Limpeza, espaço e encerramento

    **Objetivo de aprendizagem:** distinguir o que pode ser descartado do que precisa ser preservado.

    Modelos em memória ocupam VRAM; arquivos em `/content` ocupam o disco temporário. A célula libera objetos Python e mostra o tamanho persistido. Marque `APAGAR_AREA_TEMPORARIA=True` somente depois de confirmar os arquivos no Drive.

    Os dois checkpoints recentes permanecem em `runs/<run_id>/checkpoints`. O dataset compactado permanece em `data/processed`, e os WAVs comparativos em `samples`.

    **Resultado esperado:** memória CUDA liberada e um resumo final de localização e tamanho.
    """
)

code(
    """
    import gc

    APAGAR_AREA_TEMPORARIA = False
    gc.collect()
    torch.cuda.empty_cache()

    persistent_bytes = sum(
        path.stat().st_size for path in DRIVE_RUN.rglob("*") if path.is_file()
    )
    print(f"Execução persistida: {DRIVE_RUN}")
    print(f"Tamanho persistido: {persistent_bytes / 1024**3:.2f} GB")
    print(f"Dataset compactado: {archive_drive}")
    print(f"Avaliação: {evaluation_csv}")

    if APAGAR_AREA_TEMPORARIA:
        if str(WORK_ROOT).startswith("/content/FineTunning-work/") and WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)
            print("Área temporária removida.")
    else:
        print("Área temporária preservada. Ela desaparecerá quando a sessão Colab terminar.")
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
