---
title: LegacyAI
emoji: 🔄
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
---

# LegacyAI

A complete pipeline for fine-tuning a small LLM to translate COBOL code to Python.

Uses Qwen2.5-Coder-1.5B-Instruct as the base model.

## Features

- **Smart Data Selection**: Uses sentence transformers + K-Means clustering to select diverse, complex samples
- **Efficient Training**: LoRA adapters for memory-efficient fine-tuning
- **GPU Ready**: Optimized for RunPod and similar GPU cloud platforms
- **Progress Tracking**: Clear logging and progress bars throughout
- **Checkpoint Support**: Resume training from interruptions

## Requirements

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (or CPU for inference only)
- ~10GB disk space for model and checkpoints
- Streamlit (for web app)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python select_data.py --input data/cobol_python_pairs.json --output data/processed
```

This will:
- Create embeddings for all samples
- Cluster into 10 groups
- Select top 40% most complex from each cluster
- Split 90/10 into train/val
- Save as conversational format

### 3. Train the Model

```bash
python train.py --data-dir data/processed --output-dir checkpoints --epochs 3
```

Training options:
```bash
python train.py \
    --data-dir data/processed \
    --output-dir checkpoints \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --learning-rate 2e-4 \
    --lora-rank 16 \
    --max-length 2048
```

To resume from a checkpoint:
```bash
python train.py --resume checkpoints/checkpoint-500
```

### 4. Test the Model

Run example translation:
```bash
python test.py --model checkpoints/final --example
```

Translate a file:
```bash
python test.py --model checkpoints/final --input myprogram.cob --output myprogram.py
```

Interactive mode:
```bash
python test.py --model checkpoints/final --interactive
```

### 5. Run the Web App

Launch the Streamlit web interface:
```bash
streamlit run app.py
```

The web app provides:
- Side-by-side COBOL input and Python output
- Syntax highlighting and copy functionality
- Cached model loading for fast subsequent translations

## RunPod Setup

1. **Create a pod** with an NVIDIA GPU (RTX 3090/4090 or A100 recommended)

2. **SSH into the pod** and clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/legacyAI.git
cd legacyAI
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Upload your data** (cobol_python_pairs.json) to `data/`:
```bash
mkdir -p data
# Use scp, rsync, or RunPod's file upload to transfer your data
```

5. **Run the pipeline**:
```bash
# Prepare data
python select_data.py --input data/cobol_python_pairs.json --output data/processed

# Train
python train.py --data-dir data/processed --output-dir checkpoints --epochs 3

# Test
python test.py --model checkpoints/final --example
```

6. **Download results** - copy `checkpoints/final/` back to your local machine

## Hugging Face Spaces Deployment

Deploy the web app to Hugging Face Spaces:

1. **Upload your trained model to Hugging Face Hub**:
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/legacyAI checkpoints/final .
```

2. **Create a new Space** at [huggingface.co/spaces](https://huggingface.co/spaces):
   - Choose **Streamlit** as the SDK
   - Select appropriate hardware (CPU or GPU)

3. **Clone and push files to your Space**:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/legacyAI
cp app.py test.py requirements.txt README.md YOUR_SPACE_DIR/
cd YOUR_SPACE_DIR
git add . && git commit -m "Initial commit" && git push
```

4. **Set the MODEL_PATH environment variable** in Space settings:
   - Go to Settings → Variables and secrets
   - Add: `MODEL_PATH` = `YOUR_USERNAME/legacyAI`

The app will automatically load the model from Hugging Face Hub.

## Configuration Options

### Data Selection (`select_data.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `data/cobol_python_pairs.json` | Input JSON file |
| `--output` | `data/processed` | Output directory |
| `--n-clusters` | `10` | Number of K-Means clusters |
| `--top-percentage` | `0.4` | % of complex samples per cluster |
| `--train-ratio` | `0.9` | Train/val split ratio |

### Training (`train.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `data/processed` | Processed data directory |
| `--output-dir` | `checkpoints` | Checkpoint directory |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `4` | Per-device batch size |
| `--gradient-accumulation` | `4` | Gradient accumulation steps |
| `--learning-rate` | `2e-4` | Learning rate |
| `--lora-rank` | `16` | LoRA rank (lower = less memory) |
| `--max-length` | `2048` | Max sequence length |
| `--resume` | `None` | Checkpoint to resume from |

### Inference (`test.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `checkpoints/final` | Model path |
| `--device` | `auto` | Device (auto/cuda/mps/cpu) |
| `--temperature` | `0.1` | Sampling temperature |
| `--max-tokens` | `1024` | Max generation tokens |

## Memory Requirements

| Configuration | GPU Memory | Notes |
|---------------|------------|-------|
| batch_size=4, lora_rank=16 | ~14GB | Default settings |
| batch_size=2, lora_rank=16 | ~10GB | Reduced batch |
| batch_size=2, lora_rank=8 | ~8GB | Low memory GPU |
| batch_size=1, lora_rank=4 | ~6GB | Minimal |

## Output Structure

```
checkpoints/
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
└── final/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── training_info.json
```

## Troubleshooting

### Out of Memory

1. Reduce batch size: `--batch-size 2`
2. Increase gradient accumulation: `--gradient-accumulation 8`
3. Reduce LoRA rank: `--lora-rank 8`
4. Reduce max length: `--max-length 1024`

### Training Interrupted

Resume from the latest checkpoint:
```bash
python train.py --resume checkpoints/checkpoint-XXX
```

### CUDA Out of Memory on Inference

Use CPU for testing:
```bash
python test.py --model checkpoints/final --device cpu
```

## Model Architecture

- **Base Model**: Qwen/Qwen2.5-Coder-1.5B-Instruct (1.5B parameters)
- **LoRA Targets**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable Parameters**: ~0.5% of total (with default LoRA rank)

## License

This pipeline is for educational and research purposes. The base model (Qwen2.5-Coder) has its own license terms.
