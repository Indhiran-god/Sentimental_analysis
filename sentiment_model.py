import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import config

class SentimentBERT(nn.Module):
    def __init__(self, model_name=config.Config.MODEL_NAME, num_classes=config.Config.NUM_CLASSES):
        super(SentimentBERT, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Classification head
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize weights for the classifier
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        """Initialize the weights of the classification head"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for padding
            labels: Ground truth labels for training
        
        Returns:
            outputs: Model outputs including logits and loss
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False
        )
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }
    
    def predict(self, input_ids, attention_mask=None):
        """
        Make predictions without calculating loss
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for padding
        
        Returns:
            predictions: Predicted class probabilities and labels
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(logits, dim=-1)
            
            return {
                'logits': logits,
                'probabilities': probabilities,
                'predicted_labels': predicted_labels
            }

class SentimentModel:
    def __init__(self, model_name=config.Config.MODEL_NAME, num_classes=config.Config.NUM_CLASSES):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize_model(self):
        """Initialize the sentiment analysis model"""
        self.model = SentimentBERT(self.model_name, self.num_classes)
        self.model.to(self.device)
        return self.model
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'model_name': self.model_name,
                    'num_classes': self.num_classes
                }
            }, path)
            print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = SentimentBERT(
            checkpoint['model_config']['model_name'],
            checkpoint['model_config']['num_classes']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {path}")
        return self.model
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"Model: {self.model_name}")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Number of classes: {self.num_classes}")
            print(f"Device: {self.device}")

if __name__ == "__main__":
    # Test the model initialization
    sentiment_model = SentimentModel()
    model = sentiment_model.initialize_model()
    sentiment_model.get_model_summary()