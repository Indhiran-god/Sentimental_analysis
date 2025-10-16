import torch
from transformers import AutoTokenizer
import numpy as np
from sentiment_model import SentimentModel
import config

class SentimentPredictor:
    def __init__(self, model_path=None):
        self.config = config.Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        # Initialize model
        self.model_wrapper = SentimentModel()
        self.model = None
        
        if model_path:
            self.load_model(model_path)
        else:
            # Try to load the best model by default
            try:
                self.load_model(f"{self.config.MODEL_SAVE_PATH}/best_model.pt")
            except:
                print("Warning: No pre-trained model found. Please train the model first or provide a model path.")
    
    def load_model(self, model_path):
        """Load a trained model"""
        self.model_wrapper.load_model(model_path)
        self.model = self.model_wrapper.model
        print(f"Model loaded from {model_path}")
    
    def preprocess_text(self, text):
        """Preprocess input text for the model"""
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        return encoding
    
    def predict_single(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            dict: Prediction results including label, confidence, and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a trained model first.")
        
        # Preprocess the text
        encoding = self.preprocess_text(text)
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model.predict(input_ids, attention_mask)
            
            # Get results
            predicted_label = outputs['predicted_labels'].item()
            probabilities = outputs['probabilities'].cpu().numpy()[0]
            confidence = probabilities[predicted_label]
            
            # Convert to human-readable format
            sentiment_label = self.config.LABEL_MAP[predicted_label]
            
            # Create probability distribution
            prob_distribution = {
                self.config.LABEL_MAP[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        
        return {
            'text': text,
            'sentiment': sentiment_label,
            'confidence': float(confidence),
            'probabilities': prob_distribution,
            'raw_label': predicted_label
        }
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (list): List of texts to analyze
        
        Returns:
            list: List of prediction results for each text
        """
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        
        return results
    
    def analyze_sentiment_distribution(self, texts):
        """
        Analyze sentiment distribution for a batch of texts
        
        Args:
            texts (list): List of texts to analyze
        
        Returns:
            dict: Sentiment distribution statistics
        """
        predictions = self.predict_batch(texts)
        
        # Count sentiment distribution
        sentiment_counts = {label: 0 for label in self.config.LABEL_MAP.values()}
        total_confidence = 0
        
        for pred in predictions:
            sentiment_counts[pred['sentiment']] += 1
            total_confidence += pred['confidence']
        
        # Calculate percentages
        total_texts = len(texts)
        sentiment_percentages = {
            sentiment: (count / total_texts) * 100 
            for sentiment, count in sentiment_counts.items()
        }
        
        avg_confidence = total_confidence / total_texts if total_texts > 0 else 0
        
        return {
            'total_texts': total_texts,
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'average_confidence': avg_confidence,
            'predictions': predictions
        }

def main():
    """Example usage of the sentiment predictor"""
    # Initialize predictor
    predictor = SentimentPredictor()
    
    if predictor.model is None:
        print("No model available for prediction. Please train the model first.")
        return
    
    # Example texts for prediction
    example_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is okay, nothing special but gets the job done.",
        "Terrible experience. The product broke after one day of use.",
        "The service was average, could be better but not bad either.",
        "Absolutely fantastic! Would highly recommend to everyone."
    ]
    
    print("Sentiment Analysis Results:")
    print("=" * 60)
    
    # Single prediction example
    single_text = "This movie is absolutely fantastic and I loved every minute of it!"
    result = predictor.predict_single(single_text)
    
    print(f"\nSingle Prediction:")
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")
    
    # Batch prediction example
    print(f"\nBatch Predictions:")
    print("-" * 40)
    
    batch_results = predictor.predict_batch(example_texts)
    for i, result in enumerate(batch_results, 1):
        print(f"{i}. {result['text'][:50]}... -> {result['sentiment']} ({result['confidence']:.4f})")
    
    # Distribution analysis
    print(f"\nSentiment Distribution Analysis:")
    print("-" * 40)
    
    distribution = predictor.analyze_sentiment_distribution(example_texts)
    print(f"Total texts analyzed: {distribution['total_texts']}")
    print(f"Sentiment counts: {distribution['sentiment_counts']}")
    print(f"Sentiment percentages: {distribution['sentiment_percentages']}")
    print(f"Average confidence: {distribution['average_confidence']:.4f}")

if __name__ == "__main__":
    main()