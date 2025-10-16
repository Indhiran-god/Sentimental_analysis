import argparse
import sys
import os
from data_preprocessing import DataPreprocessor
from train_model import Trainer
from predict_sentiment import SentimentPredictor
import config

def main():
    parser = argparse.ArgumentParser(description="BERT-based Sentiment Analysis Model")
    parser.add_argument(
        '--mode', 
        choices=['train', 'predict', 'preprocess', 'evaluate'],
        required=True,
        help='Mode to run: train, predict, preprocess, or evaluate'
    )
    parser.add_argument(
        '--text', 
        type=str,
        help='Text to analyze sentiment (for predict mode)'
    )
    parser.add_argument(
        '--file', 
        type=str,
        help='File containing texts to analyze (one per line)'
    )
    parser.add_argument(
        '--model_path', 
        type=str,
        default=f"{config.Config.MODEL_SAVE_PATH}/best_model.pt",
        help='Path to trained model (for predict mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting model training...")
        trainer = Trainer()
        trainer.train()
        
    elif args.mode == 'predict':
        predictor = SentimentPredictor(args.model_path)
        
        if args.text:
            # Single text prediction
            result = predictor.predict_single(args.text)
            print(f"\nSentiment Analysis Result:")
            print(f"Text: {result['text']}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: {result['probabilities']}")
            
        elif args.file:
            # Batch prediction from file
            if not os.path.exists(args.file):
                print(f"Error: File {args.file} not found")
                return
            
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            if not texts:
                print("Error: No valid texts found in the file")
                return
            
            print(f"Analyzing {len(texts)} texts from {args.file}...")
            results = predictor.predict_batch(texts)
            
            # Print results
            print(f"\nBatch Sentiment Analysis Results:")
            print("=" * 80)
            for i, result in enumerate(results, 1):
                print(f"{i:3d}. {result['sentiment']:8s} ({result['confidence']:.4f}): {result['text'][:60]}...")
            
            # Print summary
            distribution = predictor.analyze_sentiment_distribution(texts)
            print(f"\nSummary:")
            print(f"Total texts: {distribution['total_texts']}")
            print(f"Sentiment distribution: {distribution['sentiment_counts']}")
            print(f"Average confidence: {distribution['average_confidence']:.4f}")
            
        else:
            print("Error: Please provide either --text or --file argument for prediction")
            parser.print_help()
    
    elif args.mode == 'preprocess':
        print("Preprocessing data...")
        preprocessor = DataPreprocessor()
        datasets = preprocessor.prepare_datasets()
        
        # Show dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(datasets['train'])}")
        print(f"Validation samples: {len(datasets['validation'])}")
        print(f"Test samples: {len(datasets['test'])}")
        
        # Show class distribution
        train_df, val_df, test_df = preprocessor.load_data()
        print(f"\nTraining set class distribution:")
        preprocessor.get_class_distribution(train_df)
        print(f"\nValidation set class distribution:")
        preprocessor.get_class_distribution(val_df)
        print(f"\nTest set class distribution:")
        preprocessor.get_class_distribution(test_df)
    
    elif args.mode == 'evaluate':
        print("Evaluating model on test set...")
        predictor = SentimentPredictor(args.model_path)
        
        # Load test data
        preprocessor = DataPreprocessor()
        _, _, test_df = preprocessor.load_data()
        
        # Convert to list of texts
        test_texts = test_df['Comment'].tolist()
        true_labels = test_df['Sentiment'].tolist()
        
        # Make predictions
        predictions = predictor.predict_batch(test_texts)
        predicted_labels = [pred['raw_label'] for pred in predictions]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(
            true_labels, 
            predicted_labels, 
            target_names=[config.Config.LABEL_MAP[i] for i in range(config.Config.NUM_CLASSES)]
        ))
        
        # Plot confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[config.Config.LABEL_MAP[i] for i in range(config.Config.NUM_CLASSES)],
            yticklabels=[config.Config.LABEL_MAP[i] for i in range(config.Config.NUM_CLASSES)]
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"{config.Config.LOGS_PATH}/confusion_matrix.png")
        print(f"Confusion matrix saved to {config.Config.LOGS_PATH}/confusion_matrix.png")

def interactive_mode():
    """Interactive mode for real-time sentiment analysis"""
    print("BERT Sentiment Analysis - Interactive Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    predictor = SentimentPredictor()
    
    while True:
        text = input("\nEnter text to analyze: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        try:
            result = predictor.predict_single(text)
            print(f"\nResult: {result['sentiment']} (Confidence: {result['confidence']:.4f})")
            print(f"Probabilities: {result['probabilities']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        interactive_mode()
    else:
        main()