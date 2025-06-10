import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords
import pandas as pd
from bert_classifier import BERTClassifier
from text_classification_dataset import TextClassificationDataset
from lstm import LSTM
import torch.nn.functional as F
from model import  DisModel
from loss import ce_loss,DS_Combin
# Load data from file
train_data = pd.read_csv('data/SustODS-PT.csv')



def format_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove words that begin with @
    text = re.sub(r'@\w+', '', text)
    # Remove words that begin with #
    text = re.sub(r'\b#\w+\b', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stop words
    pt_stp_words = stopwords.words('portuguese')
    text = ' '.join([word for word in text.split() if word not in pt_stp_words])
    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)
    return text

# Apply the format_text function to the 'text' column in train_data
train_data['text'] = train_data['text'].apply(format_text)
features = ['text', 'label']

texts = train_data['text'].tolist()
labels = train_data['label'].tolist()

def train(model, time_encoder, data_loader, optimizer, scheduler, device):
    model.train()
    model_d = DisModel(2).to(device)
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        bert_outputs = model(input_ids=input_ids, attention_mask=attention_mask).to(torch.float32)
        time_outputs  =  time_encoder.forward(input_ids.to(torch.float32))
        ###
        time_to_bert = time_encoder.LSTMtoBert(time_outputs)
        bert_to_time = time_encoder.BertToLSTM(bert_outputs)
        mse_loss1 = F.mse_loss(bert_outputs, time_to_bert)
        mse_loss2 = F.mse_loss(time_outputs, bert_to_time)
        out = [bert_outputs, time_to_bert]
        ce_loss1 = 0
        alpha = dict()
        for v_num in range(2):
            alpha[v_num] = out[v_num] + 1
            ce_loss1 += ce_loss(labels, alpha[v_num], 2, epoch + 1, 20)
        alpha_a = DS_Combin(alpha)
        evidence_a = alpha_a - 1  # fusion features
        ce_loss1 += ce_loss(labels, alpha_a, 2, epoch + 1, 20)

        loss = torch.mean(ce_loss1) + 0.01 * (mse_loss1+mse_loss2)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up parameters
# 下载到bert模型到指定位置
bert_model_name = '/home/dell4/ZY/toldbr-bert-text-classification-pt-br/bert-base-portuguese-cased/'
num_classes = 2
max_length = 128
batch_size = 16
num_epochs = 2
learning_rate = 2e-5

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)
# model.load_state_dict(torch.load('bert_pt_classifier.pth'))
input_size = 128
hidden_size = 1024
num_layers  =2
output_size = 256
dropout_rate  =  0
time_encoder = LSTM(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, time_encoder, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model,  val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)

torch.save(model.state_dict(), "bert_pt_classifier.pth")