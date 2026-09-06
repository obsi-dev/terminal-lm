# TerminalLM

A command-line based local LLM that lives in your terminal and turns natural language into executable bash commands.

Are you good at writing Bash scripts? Personally, I'm not. Unfortunately, a lot of the models that _could_ run those scripts AND have access to the terminal are paid.
So I made this. Qwen2.5-3B-Instruct fine-tuned with transformer architecture and QLoRA, trained on a dataset of Linux commands. Every command is previewed in an isolated sandbox using Docker before it runs, to ensure a command doesn't break anything, and requires human confirmation before real execution. Basic TUI built using Typer.

## Setup

### Prerequisites

- Python 3.10+
- Docker (with your user in the `docker` group)
- NVIDIA GPU with 8GB+ VRAM for training and inference
- CUDA compatible PyTorch

### Installation

\`\`\` bash
git clone git@github.com:obsi-dev/terminal-lm
cd terminal-lm
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# for building the sandbox image

docker build -t terminal-lm -f sandbox/Dockerfile sandbox
\`\`\`

### Docker Permissions

If you ever hit "permission denied" related to some Docker stuff add your user to the docker group:
\`\`\`bash
sudo usermod -aG docker $USER
\`\`\`

### Training the model

Model adapter isn't included in this repo, since weights are large and regenerable
\`\`\`bash
python training/train.py
\`\`\`
Note: This takes a while, it took me around 1.5 hours on an RTX 3070 (8GB VRAM) but once done it is persistent and the only time taken is to load weights.

## Architecture

\`\`\`
app/ - CLI entry point (Typer) and inference engine
sandbox/ - Docker based sandbox
training/ - QLoRA fine tuning script
\`\`\`

```

```
