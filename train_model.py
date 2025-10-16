# train_model.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# -----------------------------
# 1. Config
# -----------------------------
MODEL_NAME = "distilbert-base-uncased"  # You can change to any HuggingFace model
NUM_LABELS = 3  # positive, negative, neutral
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LEN = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Optional: suppress torch.compile errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# -----------------------------
# 2. Dataset
# -----------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# -----------------------------
# 3. Load Data
# -----------------------------
# FIXED: Use forward slashes or raw string to avoid escape character issues
df = pd.read_csv(r"D:\sentimental analysis\dataset\archive\sentiment_data.csv")

# Print column names to verify
print("\nColumn names in CSV:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# FIXED: Adjust column names based on your actual CSV
# Common variations: 'text', 'Text', 'review', 'sentence', etc.
# and 'sentiment', 'Sentiment', 'label', 'Label', etc.
# Replace these with your actual column names:
TEXT_COLUMN = 'Comment'  # Change this to match your CSV
SENTIMENT_COLUMN = 'Sentiment'  # Change this to match your CSV

# Check if columns exist
if TEXT_COLUMN not in df.columns or SENTIMENT_COLUMN not in df.columns:
    print(f"\nERROR: Could not find columns '{TEXT_COLUMN}' or '{SENTIMENT_COLUMN}'")
    print(f"Available columns: {df.columns.tolist()}")
    print("\nPlease update TEXT_COLUMN and SENTIMENT_COLUMN variables in the code.")
    exit(1)

# Map sentiment labels to integers if needed
# If your labels are strings like 'positive', 'negative', 'neutral':
if df[SENTIMENT_COLUMN].dtype == 'object':
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df[SENTIMENT_COLUMN] = df[SENTIMENT_COLUMN].map(label_map)
else:
    df[SENTIMENT_COLUMN] = df[SENTIMENT_COLUMN].astype(int)

# Remove any missing values
df = df.dropna(subset=[TEXT_COLUMN, SENTIMENT_COLUMN])

print(f"\nDataset size: {len(df)}")
print(f"Sentiment distribution:\n{df[SENTIMENT_COLUMN].value_counts()}")

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[SENTIMENT_COLUMN])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[SENTIMENT_COLUMN])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = SentimentDataset(train_df[TEXT_COLUMN].tolist(), train_df[SENTIMENT_COLUMN].tolist(), tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_df[TEXT_COLUMN].tolist(), val_df[SENTIMENT_COLUMN].tolist(), tokenizer, MAX_LEN)
test_dataset = SentimentDataset(test_df[TEXT_COLUMN].tolist(), test_df[SENTIMENT_COLUMN].tolist(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 4. Model, Optimizer, Loss
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

# -----------------------------
# 5. Training Function
# -----------------------------
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        # Fixed: Compatible with older PyTorch versions
        if device.type == "cuda":
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# -----------------------------
# 6. Validation Function
# -----------------------------
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# -----------------------------
# 7. Training Loop
# -----------------------------
for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

# -----------------------------
# 8. Test Evaluation
# -----------------------------
test_loss, test_acc = eval_epoch(model, test_loader, criterion, DEVICE)
print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# -----------------------------
# 9. Save Model
# -----------------------------
os.makedirs("saved_model", exist_ok=True)
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
print("Model saved in 'saved_model/' folder.")