"""
Configuration file for hyperparameters and environment settings.
"""

# Environment
DT = 0.1                      # control interval (s)
TARGET_HEIGHT = 0.5           # desired layer height (mm)

# RL training
NUM_EPISODES = 500
MAX_STEPS = 50
LR = 3e-4
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 10
BATCH_SIZE = 64

# LNN (if pretraining)
LNN_INPUT_DIM = 3
LNN_HIDDEN_DIM = 32
LNN_OUTPUT_DIM = 1