"""
Snore Detection - project paths and shared settings.

Author: Hassan Kajila
"""

import os

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_DIR, 'data')

MODEL_DIR = os.path.join(_PROJECT_DIR, 'models')
RAW_DATA_DIR = os.path.join(_DATA_DIR, 'raw')
PROCESSED_DATA_FILE = os.path.join(_DATA_DIR, 'processed', 'dataset.npz')
RANDOM_SEED = 42
