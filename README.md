# Paths of Glory AI

AI for the board game *Paths of Glory* using Behavioral Cloning and AlphaZero self-play.

## 🚀 Quick Start (Conda / Cluster Setup)

To set up the environment and start training on a cluster using Conda:

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate it
conda activate pog_ai

# 3. Start Behavioral Cloning training
# Adjust batch-size based on GPU VRAM (H200: 4096, RTX 5060: 32)
python3 train_bc.py \
  --data data/training/expert_games.jsonl \
  --batch-size 128 \
  --epochs 50 \
  --checkpoint-dir checkpoints/bc
```

## 📂 Project Structure
- `src/env/`: Game environment (`PogEnv`)
- `src/rl/`: Neural network and MCTS implementation
- `train_bc.py`: Behavioral Cloning trainer
- `TRAINING_GUIDE.md`: Detailed training instructions
