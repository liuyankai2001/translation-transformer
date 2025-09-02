from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = Path(__file__).parent.parent / 'data' /'raw'
PROCESS_DATA_DIR = Path(__file__).parent.parent / 'data' /'processed'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'

# 模型参数
DIM_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

# 训练参数
SEQ_LEN = 32
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 100
ENCODER_LAYERS = 2
