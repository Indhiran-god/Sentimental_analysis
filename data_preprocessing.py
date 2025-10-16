import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import config

class DataPreprocessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.Config.MODEL_NAME)
        self.config = config.Config()
        
    def load_data(self):
        """Load and preprocess the sentiment dataset"""
        print("Loading dataset...")
        df = pd.read_csv(config.Config.DATA_PATH)
        
        # Clean the data
        df = self._clean_data(df)
        
        # Split the data
        train_df, temp_df = train_test_split(
            df, 
            test_size=config.Config.VALIDATION_SIZE + config.Config.TEST_SIZE,
            random_state=config.Config.RANDOM_SEED,
            stratify=df['Sentiment']
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=config.Config.TEST_SIZE / (config.Config.VALIDATION_SIZE + config.Config.TEST_SIZE),
            random_state=config.Config.RANDOM_SEED,
            stratify=temp_df['Sentiment']
        )
        
        print(f"Dataset sizes - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _clean_data(self, df):
        """Clean and preprocess the text data"""
        # Remove rows with missing values
        df = df.dropna(subset=['Comment', 'Sentiment'])
        
        # Ensure sentiment labels are integers
        df['Sentiment'] = df['Sentiment'].astype(int)
        
        # Filter only valid sentiment labels (0, 1, 2)
        df = df[df['Sentiment'].isin([0, 1, 2])]
        
        # Clean text - remove extra whitespace and ensure string type
        df['Comment'] = df['Comment'].astype(str).str.strip()
        
        # Remove empty comments
        df = df[df['Comment'].str.len() > 0]
        
        return df
    
    def tokenize_function(self, examples):
        """Tokenize the text data for BERT"""
        tokenized = self.tokenizer(
            examples['Comment'],
            padding='max_length',
            truncation=True,
            max_length=config.Config.MAX_LENGTH,
            return_tensors=None
        )
        return tokenized
    
    def prepare_datasets(self):
        """Prepare datasets for training"""
        train_df, val_df, test_df = self.load_data()
        
        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        # Set format for PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Sentiment'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Sentiment'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Sentiment'])
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return dataset_dict
    
    def get_class_distribution(self, df):
        """Get the distribution of sentiment classes"""
        distribution = df['Sentiment'].value_counts().sort_index()
        print("Class distribution:")
        for label, count in distribution.items():
            print(f"  {config.Config.LABEL_MAP[label]}: {count} samples")
        return distribution

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    datasets = preprocessor.prepare_datasets()
    print("Data preprocessing completed successfully!")