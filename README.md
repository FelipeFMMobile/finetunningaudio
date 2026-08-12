# FineTunning — clonagem de voz com Qwen3-TTS

Projeto didático para preparar uma gravação em português brasileiro, executar clonagem de voz *zero-shot*, fazer *fine-tuning* single-speaker do Qwen3-TTS e comparar os resultados.

O código fica neste repositório. A gravação é enviada do Mac para a sessão do Colab pelo navegador e os resultados são baixados em um pacote `.tar`.

## Notebook principal

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FelipeFMMobile/finetunningaudio/blob/main/notebooks/01_qwen3_tts_clone_finetuning.ipynb)

Como o repositório é privado, abra o notebook pelo seletor **Arquivo → Abrir notebook → GitHub** do Colab e autorize o acesso aos repositórios privados, caso o link direto solicite autenticação.

## Antes de começar

1. Leia o [guia de gravação](docs/GUIA_GRAVACAO.md).
2. Grave o texto de [`scripts/roteiro_gravacao_ptbr.txt`](scripts/roteiro_gravacao_ptbr.txt) em um único WAV.
3. Abra o notebook no Colab e escolha **Ambiente de execução → Alterar tipo de ambiente de execução → GPU**.
4. Em T4, use `INPUT_MODE = "raw"`, prepare os dados e baixe `*_dataset_preparado.tar.gz` na seção 2.4.
5. Abra uma sessão L4/A100, use `INPUT_MODE = "prepared"`, envie o pacote e continue no zero-shot e fine-tuning.
6. Ao final, aguarde o download do pacote `<run_id>.tar` antes de fechar a sessão.

## Organização dos artefatos

```text
Mac                                  Sessão Colab
├── voz_ptbr.wav ───── upload ─────► /content/FineTunning-storage/
└── <run_id>.tar ◄─── download ─────     └── <run_id>/
                                             ├── config.json
GitHub privado                              ├── data/
├── notebook                               ├── train.log
├── roteiro                                ├── checkpoints/
├── documentação                           └── samples/
└── testes
```

O painel **Arquivos** do Colab mostra o disco da máquina virtual remota, não as pastas do Mac. `/content` é temporário e pode desaparecer quando a sessão expira. O notebook mantém os dois checkpoints mais recentes e cria um pacote final para download.

## Correspondência com o curso

| Parte do notebook | Notebook de referência |
|---|---|
| Parte A — Preparação dos dados | `02_data_prep.ipynb` |
| Parte B — Clonagem zero-shot | `01_voice_cloning.ipynb` |
| Parte C — Fine-tuning e teste | `03_finetune.ipynb` |

A ordem de execução começa pela preparação dos dados porque o áudio usado no zero-shot é extraído da própria gravação.

## Estratégia automática de GPU

| GPU detectada | Modelo | Precisão | Batch inicial |
|---|---|---|---|
| T4, 16 GB | Qwen3-TTS 0.6B | FP32 | preparação e zero-shot; treino bloqueado |
| L4, 24 GB | Qwen3-TTS 1.7B | BF16 | 1 |
| A100 ou melhor | Qwen3-TTS 1.7B | BF16 | 4 |

A T4 não possui memória suficiente para o fine-tuning completo em FP32: mesmo com batch 1, o AdamW não consegue manter modelo, gradientes e estados. Ela continua útil para preparar o corpus e gerar a linha de base zero-shot. O treinamento oficial utilizado aqui requer L4/A100; o Mac M1 Pro é adequado para editar e versionar o projeto.

## Validação local

```bash
uv sync --extra dev
uv run pytest
```

Os testes validam a estrutura, a ordem didática e a sintaxe das células Python. Eles não baixam modelos nem iniciam treinamento.

## Privacidade e uso responsável

Use somente uma voz própria ou uma voz para a qual exista consentimento explícito. Não publique a voz ou o modelo sem compreender as implicações. O `.gitignore` bloqueia formatos comuns de áudio e pesos, mas revise sempre `git status` antes de cada commit.

## Referências

- [Qwen3-TTS oficial](https://github.com/QwenLM/Qwen3-TTS)
- [Fine-tuning oficial](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
- [Modelos Qwen no Hugging Face](https://huggingface.co/Qwen)

## Licença

O material autoral deste repositório usa a licença MIT. O Qwen3-TTS e seus pesos seguem as licenças publicadas por seus respectivos autores.
