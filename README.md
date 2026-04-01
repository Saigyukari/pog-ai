# Paths of Glory AI

AI for the board game *Paths of Glory* using Behavioral Cloning and AlphaZero self-play.

## 🚀 Quick Start (Cluster Setup)

To set up the environment and start training on a cluster:

```bash
# Clone the repo (if not already done)
# git clone <repo_url> && cd PoGAIV1

# Run the setup script
chmod +x setup_env.sh
./setup_env.sh

# Start Behavioral Cloning training
source venv/bin/activate
python3 train_bc.py \
  --data data/training/expert_games.jsonl \
  --batch-size 32 \
  --epochs 50
```

## 📂 Project Structure
- `src/env/`: Game environment (`PogEnv`)
- `src/rl/`: Neural network and MCTS implementation
- `train_bc.py`: Behavioral Cloning trainer
- `TRAINING_GUIDE.md`: Detailed training instructions
