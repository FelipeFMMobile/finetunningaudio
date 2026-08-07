# FineTunning — clonagem de voz com Qwen3-TTS

Projeto didático para preparar uma gravação em português brasileiro, executar clonagem de voz *zero-shot*, fazer *fine-tuning* single-speaker do Qwen3-TTS e comparar os resultados.

O código fica neste repositório. Por padrão, a gravação é enviada do Mac para a sessão do Colab pelo navegador e os resultados são baixados em um pacote `.tar`. O Google Drive permanece disponível como modo opcional.

## Notebook principal

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FelipeFMMobile/finetunningaudio/blob/main/notebooks/01_qwen3_tts_clone_finetuning.ipynb)

Como o repositório é privado, abra o notebook pelo seletor **Arquivo → Abrir notebook → GitHub** do Colab e autorize o acesso aos repositórios privados, caso o link direto solicite autenticação.

## Antes de começar

1. Leia o [guia de gravação](docs/GUIA_GRAVACAO.md).
2. Grave o texto de [`scripts/roteiro_gravacao_ptbr.txt`](scripts/roteiro_gravacao_ptbr.txt) em um único WAV.
3. Abra o notebook no Colab e escolha **Ambiente de execução → Alterar tipo de ambiente de execução → GPU**.
4. Execute as células em ordem. Quando solicitado, escolha `voz_ptbr.wav` no seletor de arquivos do navegador.
5. Ao final, aguarde o download do pacote `<run_id>.tar` antes de fechar a sessão.

## Organização dos artefatos

```text
Mac                                  Sessão Colab
├── voz_ptbr.wav ───── upload ─────► /content/FineTunning-storage/
└── <run_id>.tar ◄─── download ───── ├── data/
                                     ├── exports/
GitHub privado                       └── runs/<run_id>/
├── notebook                             ├── config.json
├── roteiro                               ├── metrics.jsonl
├── documentação                         ├── checkpoints/
└── testes                               └── samples/
```

O painel **Arquivos** do Colab mostra o disco da máquina virtual remota, não as pastas do Mac. `/content` é temporário e pode desaparecer quando a sessão expira. O notebook mantém os dois checkpoints mais recentes e cria um pacote final para download. Para persistência automática, altere `STORAGE_MODE = "upload"` para `STORAGE_MODE = "drive"` na seção 3.

Para restaurar um pacote baixado anteriormente, defina `RESTAURAR_PACOTE_ANTERIOR = True`, envie o `.tar` quando solicitado e preencha `RUN_ID_OVERRIDE` com o identificador da execução restaurada.

## Estratégia automática de GPU

| GPU detectada | Modelo | Precisão | Batch inicial |
|---|---|---|---|
| T4, 16 GB | Qwen3-TTS 0.6B | FP16 | 1 |
| L4, 24 GB | Qwen3-TTS 1.7B | BF16 | 2 |
| A100 ou melhor | Qwen3-TTS 1.7B | BF16 | 4 |

A escolha pode ser sobrescrita na célula central de configuração. O Mac M1 Pro é adequado para editar e versionar o projeto, mas o treinamento oficial utilizado aqui requer uma GPU NVIDIA com CUDA.

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
