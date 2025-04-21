!pip install -q trl accelerate

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from trl import PPOTrainer, PPOConfig
from datasets import Dataset
import torch

# Load GPT-2 and tokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_model.to(device)

# Define reward function based on phishing-like traits
def phishing_reward_fn(text):
    reward = 0
    text_lower = text.lower()
    if "account" in text_lower: reward += 0.3
    if "click here" in text_lower: reward += 0.3
    if "verify" in text_lower: reward += 0.2
    if "suspended" in text_lower or "urgent" in text_lower: reward += 0.2
    return reward

# Prepare dataset with prompts
prompts = ["Urgent: Your account has been"] * 20
data = Dataset.from_dict({"prompt": prompts})

# PPO config and trainer
ppo_config = PPOConfig(model_name="gpt2", batch_size=4)
ppo_trainer = PPOTrainer(model=gpt_model, tokenizer=gpt_tokenizer, config=ppo_config)

# Training loop with PPO
trained_phish = []
for batch in data["prompt"]:
    input_ids = gpt_tokenizer(batch, return_tensors="pt").input_ids.to(device)
    response = gpt_model.generate(input_ids, max_length=80)
    decoded = gpt_tokenizer.decode(response[0], skip_special_tokens=True)
    reward = phishing_reward_fn(decoded)
    ppo_trainer.step([batch], [decoded], [reward])
    trained_phish.append(decoded)

# Add generated samples to training
texts.extend(trained_phish)
labels.extend([1] * len(trained_phish))

# Re-tokenize and retrain as before
new_train_enc = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
new_train_dataset = EmailDataset(new_train_enc, labels)
new_train_loader = DataLoader(new_train_dataset, batch_size=8, shuffle=True)

model.train()
for epoch in range(2):
    total_loss = 0
    for batch in new_train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"RLHF Adversarial Epoch {epoch+1}, Loss: {total_loss:.4f}")
from transformers import GPT2Tokenizer, GPT2LMHeadModel

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_model.to(device)

generated_phish = []
for _ in range(20):
    prompt = "Urgent: Account suspended due to"
    inputs = gpt_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = gpt_model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)
    new_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_phish.append(new_text)

# Add generated samples to training
texts.extend(generated_phish)
labels.extend([1]*len(generated_phish))

# Re-tokenize and prepare training data
new_train_enc = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
new_train_dataset = EmailDataset(new_train_enc, labels)
new_train_loader = DataLoader(new_train_dataset, batch_size=8, shuffle=True)

# Retrain model for few more epochs with augmented data
model.train()
for epoch in range(2):
    total_loss = 0
    for batch in new_train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Adversarial Epoch {epoch+1}, Loss: {total_loss:.4f}")
