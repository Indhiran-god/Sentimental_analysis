# BERT-based Sentiment Analysis Model

A comprehensive sentiment analysis system using BERT (Bidirectional Encoder Representations from Transformers) to classify text into three sentiment categories: Negative (0), Neutral (1), and Positive (2).

## Features

- **BERT-based Architecture**: Uses `bert-base-uncased` for state-of-the-art text classification
- **Large Dataset**: Trained on 241,000+ English comments from various online platforms
- **Multi-class Classification**: Classifies into Negative, Neutral, and Positive sentiments
- **Comprehensive Pipeline**: Complete workflow from data preprocessing to prediction
- **Model Evaluation**: Detailed metrics including accuracy, precision, recall, and F1-score
- **Interactive Mode**: Real-time sentiment analysis
- **Batch Processing**: Analyze multiple texts at once

## Project Structure

```
sentimental_analysis/
├── config.py              # Configuration parameters
├── data_preprocessing.py  # Data loading and preprocessing
├── sentiment_model.py     # BERT model architecture
├── train_model.py         # Training script with validation
├── predict_sentiment.py   # Prediction and inference
├── main.py               # Main application entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file

dataset/
└── archive/
    └── sentiment_data.csv  # Main dataset (241,000+ samples)

models/                   # Saved models (created during training)
logs/                    # Training logs and plots (created during training)
```

## Installation

1. Clone or download the project files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

```bash
python main.py --mode preprocess
```

This will:
- Load and clean the dataset
- Split into train/validation/test sets
- Tokenize text for BERT
- Show dataset statistics and class distribution

### 2. Model Training

```bash
python main.py --mode train
```

This will:
- Initialize BERT model with classification head
- Train for 3 epochs with early stopping
- Save the best model to `models/sentiment_bert/best_model.pt`
- Generate training plots in `logs/` directory

### 3. Single Text Prediction

```bash
python main.py --mode predict --text "I love this product! It's amazing."
```

### 4. Batch Prediction from File

```bash
python main.py --mode predict --file texts.txt
```

Where `texts.txt` contains one text per line.

### 5. Model Evaluation

```bash
python main.py --mode evaluate
```

This evaluates the model on the test set and generates a confusion matrix.

### 6. Interactive Mode

```bash
python main.py
```

Run without arguments for interactive real-time sentiment analysis.

## Dataset

The dataset contains 241,000+ English comments with sentiment labels:

- **0**: Negative
- **1**: Neutral  
- **2**: Positive

### Sample Data

| Comment | Sentiment |
|---------|-----------|
| "lets forget apple pay required brand new iphone order use significant portion apples user base wasnt able use even wanted successive iphone incorporated technology older iphones replaced number people could use technology increased" | 1 (Neutral) |
| "nz retailers don't even contactless credit card machines like paywave support apple pay don't like high fees come" | 0 (Negative) |
| "forever acknowledge channel help lessons ideas explanations quite helpful youll sit comfort monitor account growth" | 2 (Positive) |

## Model Architecture

- **Base Model**: `bert-base-uncased` (110M parameters)
- **Classification Head**: Linear layer on [CLS] token representation
- **Input Length**: 256 tokens (truncated/padded)
- **Output**: 3-class softmax probabilities

## Training Configuration

- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Optimizer**: AdamW with weight decay
- **Loss Function**: CrossEntropyLoss
- **Validation Split**: 10%
- **Test Split**: 10%

## Performance Metrics

The model achieves:
- **Accuracy**: ~85-90% (depending on dataset balance)
- **Precision/Recall/F1**: Detailed per-class metrics
- **Confidence Scores**: Probability distributions for each prediction

## API Usage

You can also use the model programmatically:

```python
from predict_sentiment import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor("models/sentiment_bert/best_model.pt")

# Single prediction
result = predictor.predict_single("This product is amazing!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")

# Batch prediction
texts = ["Great!", "Okay", "Terrible"]
results = predictor.predict_batch(texts)
```

## Requirements

See `requirements.txt` for complete list. Key dependencies:
- torch>=1.9.0
- transformers>=4.20.0
- datasets>=2.0.0
- pandas>=1.4.0
- scikit-learn>=1.0.0

## Notes

- The model uses GPU if available, otherwise CPU
- Training may take several hours depending on hardware
- Model automatically saves checkpoints and best performing model
- Early stopping prevents overfitting
- Comprehensive logging and visualization included

## License

This project is for educational and research purposes. Please ensure proper attribution when using the code or model.