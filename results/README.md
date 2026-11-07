# Model Evaluation Results

This folder contains the evaluation results of the Bank Complaint Classification project.

## Model Comparison
- final_model_comparison.csv: Performance comparison of all trained models.
- final_summary.json: Final model comparison summary.
- model_comparison.png: Model performance visualization.

## Error Analysis
- confusion_pairs.csv: Most frequent classification confusion pairs.
- errors_by_class.csv: Error distribution across complaint categories.

## Best Model: DistilBERT (384 Tokens)
The best-performing model was DistilBERT with a maximum sequence length of 384 tokens.

Evaluation metrics:
- Accuracy: 0.8654
- Macro Precision: 0.8453
- Macro Recall: 0.8666
- Macro F1-score: 0.8552
- Weighted F1-score: 0.8662

The best_model folder contains its classification report,
confusion matrix, and evaluation metrics.
