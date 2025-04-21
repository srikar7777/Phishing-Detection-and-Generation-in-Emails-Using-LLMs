!pip uninstall -y transformers

!pip install transformers --no-cache-dir

import transformers
print(transformers.__version__)
!pip install kaggle kagglehub -q

import os
os.environ['KAGGLE_USERNAME'] = "srikarmaddena"
os.environ['KAGGLE_KEY'] = "********************************"

import kagglehub
phish_path = kagglehub.dataset_download("oakent/phishing-emails-mbox")
enron_path = kagglehub.dataset_download("wcukierski/enron-email-dataset")

print("Phishing path:", phish_path)
print("Enron (ham) path:", enron_path)

import mailbox
import os
import email

vtriad_tags = []  

def extract_plain_text(msg):
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            return msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    except:
        return ""

phishing_path = "/kaggle/input/phishing-emails-mbox/phishing3.mbox"
phish_mbox = mailbox.mbox(phishing_path)
phish_texts = []
for msg in phish_mbox:
    text = extract_plain_text(msg)
    if text:
        phish_texts.append(text)
        vtriad_tags.append(("email", "user", "social engineering"))  # Example V-Triad tagging


import pandas as pd
ham_df = pd.read_csv("/kaggle/input/enron-email-dataset/emails.csv")
ham_texts = ham_df['message'].dropna().tolist()
ham_texts = ham_texts[:len(phish_texts)]


texts = phish_texts + ham_texts
labels = [1]*len(phish_texts) + [0]*len(ham_texts)

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

#Tokenization using RoBERTa
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, train_labels)
val_dataset = EmailDataset(val_encodings, val_labels)

#RoBERTa Model & Extended Training Loop
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = RobertaForSequenceClassification.from_pretrained("roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(5):
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


model.eval()
preds, true_labels, pred_probs = [], [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        pred_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print("\nBasic Classification Report:")
print(classification_report(true_labels, preds))

cm = confusion_matrix(true_labels, preds)
ConfusionMatrixDisplay(cm, display_labels=["Ham", "Phish"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


from bert_score import score as bertscore
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import corpus_bleu

pred_texts = ["phishing" if p == 1 else "ham" for p in preds]
true_texts = ["phishing" if l == 1 else "ham" for l in true_labels]

P, R, F1 = bertscore(pred_texts, true_texts, lang="en")
print("\nBERTScore:")
print("Precision:", P.mean().item())
print("Recall:", R.mean().item())
print("F1:", F1.mean().item())

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_results = [scorer.score(ref, pred) for ref, pred in zip(true_texts, pred_texts)]
print("\nROUGE (avg):")
print({key: sum(d[key].fmeasure for d in rouge_results)/len(rouge_results) for key in rouge_results[0]})

bleu_input = [[ref.split()] for ref in true_texts]
bleu_preds = [pred.split() for pred in pred_texts]
print("\nBLEU:")
print("Score:", corpus_bleu(bleu_input, bleu_preds))
