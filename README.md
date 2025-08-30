# cyberbullying-detection-system
# Cyberbullying Detection Model (DistilBERT)

This project is a **binary text classification model** that detects **Cyberbullying vs Normal** text.  
It is fine-tuned on a dataset of offensive/profane/abusive language to help identify harmful online content.

---

## Model on Hugging Face

[AfreenT/cyberbullying-model](https://huggingface.co/AfreenT/cyberbullying-model)

---

## Usage

You can directly use this model with Transformers:

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load model & tokenizer
model = DistilBertForSequenceClassification.from_pretrained("AfreenT/cyberbullying-model")
tokenizer = DistilBertTokenizer.from_pretrained("AfreenT/cyberbullying-model")

# Example
text = "You are a nigga!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()

label = "Cyberbullying" if prediction == 1 else "Normal"
print("Prediction:", label)


## Web App (Flask + HTML)

The model is deployed using Flask API + Frontend (HTML/JS).

**API endpoint:** `/predict`  

**Input:**  
```json
{"text": "your message"}

"Cyberbullying" or "Normal"


## Tech Stack

Python

PyTorch

Hugging Face Transformers

Flask

HTML/CSS/JS (Frontend)
