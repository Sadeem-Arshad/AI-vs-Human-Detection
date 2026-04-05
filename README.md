#  AI vs Human Text Detection

A comprehensive NLP study comparing 8 models for detecting whether 
text is AI-generated or human-written, using the Kaggle AI-vs-Human 
dataset (20,000 balanced samples).

##  Dataset
- **Source**: Kaggle — AI vs Human Text (`shanegerami/ai-vs-human-text`)
- **Size**: 10,000 AI + 10,000 Human samples (balanced)
- **Preprocessing**: Lowercasing, special character removal, 
  contraction fixing, null filtering

##  Models Compared
| Model | Accuracy |
|---|---|
| DistilBERT (fine-tuned) | 97.38% |
| MLP + TF-IDF | 95.15% |
| CNN + TF-IDF | 95.00% |
| XGBoost + Sentence Transformers | 93.00% |
| T5 | 89.56% |
| RoBERTa | 88.50% |
| Ensemble (BERT + Cosine Similarity) | 76.50% |
| Sentence Transformers + Cosine Similarity | 69.00% |

##  Key Features
- **Embeddings**: SentenceTransformer `all-MiniLM-L6-v2` 
  (batched encoding, 512/batch)
- **Transformers**: DistilBERT, RoBERTa, T5 fine-tuned with 
  HuggingFace Trainer API
- **Classical Models**: TF-IDF + CNN/MLP with PyTorch, 
  XGBoost on semantic embeddings
- **Explainability**: SHAP token-level interpretation on 
  DistilBERT predictions
- **GPU-accelerated** training with mixed precision (AMP + GradScaler)

##  Tech Stack
Python, PyTorch, HuggingFace Transformers, Sentence-Transformers,
XGBoost, Scikit-learn, SHAP, Matplotlib, Pandas

##  How to Run
1. Clone the repo
2. `pip install -r requirements.txt`
3. Run `Human_and_AI_text_identification.ipynb` in Jupyter/Colab
