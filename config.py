import os

class Config:
    # Data paths
    DATA_PATH = "dataset/archive/sentiment_data.csv"
    MODEL_SAVE_PATH = "models/sentiment_bert"
    LOGS_PATH = "logs"
    
    # Model parameters
    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    NUM_CLASSES = 3  # Negative (0), Neutral (1), Positive (2)
    
    # Training parameters
    TRAIN_SIZE = 0.8
    VALIDATION_SIZE = 0.1
    TEST_SIZE = 0.1
    RANDOM_SEED = 42
    
    # Labels mapping
    LABEL_MAP = {
        0: "Negative",
        1: "Neutral", 
        2: "Positive"
    }
    
    # Create directories if they don't exist
    @staticmethod
    def create_directories():
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(Config.LOGS_PATH, exist_ok=True)