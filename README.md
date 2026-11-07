# Bank Complaint Classification

An NLP-based project for classifying bank customer complaints into five categories.

## Project Overview
The project compares recurrent neural networks and Transformer-based models.

## Complaint Categories
- credit_card
- credit_reporting
- debt_collection
- mortgages_and_loans
- retail_banking

## Models Evaluated
- SimpleRNN
- LSTM
- GRU
- DistilBERT (256 tokens)
- DistilBERT (384 tokens)

## Model Comparison
| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| DistilBERT (384) | 86.54% | 85.52% | 86.62% |
| DistilBERT (256) | 85.74% | 84.89% | 85.89% |
| GRU | 83.06% | 82.14% | 83.27% |
| LSTM | 81.95% | 80.69% | 82.15% |
| SimpleRNN | 72.45% | 67.56% | 72.15% |

## Best Model
**DistilBERT (384 tokens)**

- Accuracy: 86.54%
- Macro Precision: 84.53%
- Macro Recall: 86.66%
- Macro F1: 85.52%
- Weighted F1: 86.62%

## Error Analysis
- Test samples: 18,645
- Correct predictions: 16,135
- Incorrect predictions: 2,510
- Error rate: 13.46%

The most frequent confusion involved credit_reporting and debt_collection.

## Technologies
- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Scikit-learn
- Pandas
- Gradio

## Project Structure
```text
Bank-Complaint-Classification/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   └── DistilBERT_384/
└── results/
```

## Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Place the trained model files inside `model/DistilBERT_384/`.
3. Run: `python app.py`

Required model files:
- config.json
- model.safetensors
- tokenizer.json
- tokenizer_config.json
- label_encoder.pkl

## Disclaimer
This project is for educational and research purposes. Predictions are decision-support outputs, not definitive financial judgments.