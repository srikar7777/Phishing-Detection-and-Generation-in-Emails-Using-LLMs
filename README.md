#Phishing Email Detection & Adversarial Generation using LLMs

This project presents a dual-function system for email security:  
1. **Phishing Detection** using a fine-tuned RoBERTa model  
2. **Phishing Email Generation** using GPT-2 enhanced with Reinforcement Learning with Human Feedback (RLHF)

By combining these two components, the system not only identifies malicious messages but also simulates new phishing threats for adversarial training, making the detector more robust and adaptive.

---
##Repository Structure

```
.
├── phsDetection.py       
├── phsGeneration.py      
├── datasets/             # indivudial phishing datasets of (.mbox) and ham (CSV) emails
├── README.md             
```
---

## Features

-  Fine-tuned **RoBERTa** classifier for binary phishing detection  
-  **GPT-2** based generation of phishing-style text using prompts  
-  Reward function tailored to phishing indicators (e.g. “click here”, “verify”)  
-  Integration of synthetic emails back into training (adversarial retraining)  
-  Evaluation using metrics like Accuracy, F1, BLEU, ROUGE, and BERTScore  

---

##  Getting Started

Follow these steps to clone, install dependencies, and run the project.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/phishing-llm-project.git
cd phishing-llm-project
```

### 2. Install Dependencies

We recommend Python 3.8+ and a virtual environment.

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install transformers datasets trl accelerate torch
```

---

##  How to Use

### 🿦 Run Phishing Detection

```bash
python phsDetection.py
```

This will:
- Load and preprocess the phishing + ham datasets  
- Fine-tune a RoBERTa classifier  
- Evaluate performance (Accuracy, F1, Confusion Matrix)  

### 🿧 Run Phishing Email Generation

```bash
python phsGeneration.py
```

This will:
- Use GPT-2 to generate phishing-style emails from prompt
- Apply a reward function based on phishing traits
- PPO (RLHF) to optimize outputs
- Print and store filtered, high-confidence phishing emails

These emails are then injected into the training set for retraining the classifier.


---

## 🔍 Use Cases

- Research in adversarial robustness of NLP models
- Training phishing-aware security tools
- Simulating phishing campaigns for defense system testing

---

##  Author

**Srikar Maddena**  


---

## 📃 License

This project is released for educational and research purposes only. Please use responsibly.
