# Bank Complaint Classification

An NLP-based machine learning project that automatically classifies customer bank complaints into five categories using Recurrent Neural Networks and Transformer-based models.

## Project Overview

This project aims to build an intelligent text classification system that identifies the appropriate category of a bank customer's complaint based on its textual content.

Five different deep learning models were trained and evaluated to compare their classification performance and identify the best-performing model.

The final application uses **DistilBERT with a maximum sequence length of 384 tokens**, integrated into an interactive Gradio interface.
<img width="1169" height="812" alt="Screenshot 2026-11-07 175715" src="https://github.com/user-attachments/assets/f897dbbb-dff2-4115-9034-4efb002c6dcb" />
<img width="1157" height="855" alt="Screenshot 2026-11-07 175638" src="https://github.com/user-attachments/assets/7f845bce-6730-47aa-b770-8febf5e98314" />

<img width="951" height="839" alt="Screenshot 2026-11-07 175603" src="https://github.com/user-attachments/assets/c52df43d-7c91-43d4-b3b6-4bdce90d8743" />


## Complaint Categories

The model classifies complaints into five categories:

- `credit_card`
- `credit_reporting`
- `debt_collection`
- `mortgages_and_loans`
- `retail_banking`

## Models Evaluated

The following models were implemented and compared:

1. SimpleRNN
2. LSTM
3. GRU
4. DistilBERT (256 tokens)
5. DistilBERT (384 tokens)

## Model Performance Comparison

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| DistilBERT (384) | **86.54%** | **85.52%** | **86.62%** |
| DistilBERT (256) | 85.74% | 84.89% | 85.89% |
| GRU | 83.06% | 82.14% | 83.27% |
| LSTM | 81.95% | 80.69% | 82.15% |
| SimpleRNN | 72.45% | 67.56% | 72.15% |

## Best Performing Model

**DistilBERT (384 Tokens)** achieved the best overall performance.

| Metric | Score |
|---|---:|
| Accuracy | 86.54% |
| Macro Precision | 84.53% |
| Macro Recall | 86.66% |
| Macro F1-score | 85.52% |
| Weighted F1-score | 86.62% |

The 384-token configuration performed better than the 256-token configuration, indicating that retaining more textual context helped improve classification performance.

## Error Analysis

The best model was evaluated on **18,645 test samples**.

| Evaluation | Count |
|---|---:|
| Total Test Samples | 18,645 |
| Correct Predictions | 16,135 |
| Incorrect Predictions | 2,510 |
| Error Rate | 13.46% |

The analysis identified frequent confusion between:

- `credit_reporting` and `debt_collection`
- `credit_reporting` and `mortgages_and_loans`
- `credit_reporting` and `credit_card`

These errors may be related to overlapping complaint topics, insufficient context, and similarities in customer narratives.

## Technologies & Tools

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
│
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
│
├── model/
│   └── DistilBERT_384/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── label_encoder.pkl
│
└── results/
    ├── final_model_comparison.csv
    ├── final_summary.json
    ├── model_comparison.png
    ├── confusion_pairs.csv
    ├── errors_by_class.csv
    │
    └── best_model/
        ├── classification_report.csv
        ├── confusion_matrix.csv
        ├── confusion_matrix.png
        └── metrics.json
```

## Model Download

The trained model is approximately 255 MB and is distributed separately through GitHub Releases.

**Download:** [DistilBERT 384 – v1.0.0]([ADD_RELEASE_LINK_HERE](https://github.com/hassansaleh70/Bank-Complaint-Classification/releases/tag/v1.0.0))

After downloading:

1. Download `model.zip` from the Releases section.
2. Extract the ZIP file into the project root directory.
3. Make sure the extracted structure is:

```text
model/
└── DistilBERT_384/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── label_encoder.pkl
```

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/hassansaleh70/Bank-Complaint-Classification.git
cd Bank-Complaint-Classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Trained Model

Download `model.zip` from GitHub Releases and extract it into the project root directory.

### 4. Run the Application

```bash
python app.py
```

The Gradio application will start locally. Open the URL displayed in your terminal to interact with the classifier.

## Results & Evaluation

The `results/` directory contains:

- Model performance comparison.
- Classification metrics.
- Confusion matrix.
- Error analysis by class.
- Frequent classification confusion pairs.

These evaluation artifacts provide insights into model performance and classification limitations.

## Disclaimer

This project is developed for educational and research purposes.

Predictions are intended to support complaint classification and should not be considered definitive financial judgments or a replacement for human review.
