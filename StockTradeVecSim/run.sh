#!/bin/bash

PYTHON_SCRIPT="ensemble.py"

# python $PYTHON_SCRIPT --policy "PPO" --policies "PPO"
# python $PYTHON_SCRIPT --policy "SAC" --policies "SAC"
# python $PYTHON_SCRIPT --policy "DDPG" --policies "DDPG"

# python $PYTHON_SCRIPT --policy "Ensemble" --policies "PPO" "SAC" "DDPG"

# python $PYTHON_SCRIPT --policy "Ensemble_5" --policies "PPO" "PPO" "PPO" "PPO" "PPO" "SAC" "SAC" "SAC" "SAC" "SAC" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG"

# python $PYTHON_SCRIPT --policy "Ensemble_10" --policies "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG"

python $PYTHON_SCRIPT --policy "Ensemble_20" --policies "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "PPO" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "SAC" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG" "DDPG"
