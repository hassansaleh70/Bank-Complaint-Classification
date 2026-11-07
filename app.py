import os
import html
import torch
import gradio as gr
import joblib

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =========================================================
# 1. MODEL LOADING
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model", "DistilBERT_384")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(device)

model.eval()

label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

print(f"Model loaded successfully on {device}")
print("Classes:", list(label_encoder.classes_))


# =========================================================
# 2. PREDICTION FUNCTION
# =========================================================

def predict_complaint(text):
    if not text or not text.strip():
        return {}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=384
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)

    probabilities = probabilities.cpu().numpy()[0]

    results = {
        label: float(probability)
        for label, probability in zip(
            label_encoder.classes_,
            probabilities
        )
    }

    return dict(
        sorted(
            results.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


# =========================================================
# 3. CUSTOM CSS
# =========================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #071329;
    --card: #10213e;
    --card2: #142a4c;
    --blue: #1685f8;
    --cyan: #38bdf8;
    --green: #22df8a;
    --red: #ef4444;
    --text: #f8fafc;
    --muted: #a9bddb;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
    padding: 18px 24px !important;
}

.gradio-container,
.gradio-container .main,
.gradio-container .wrap {
    background: var(--bg) !important;
}

footer {
    display: none !important;
}

/* HEADER */

#main-header {
    background: linear-gradient(110deg, #102a50, #071a38);
    border: 1px solid #174b86;
    border-radius: 20px;
    padding: 25px 30px;
    margin-bottom: 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,.2);
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 20px;
}

.header-icon {
    width: 78px;
    height: 78px;
    border-radius: 18px;
    background: linear-gradient(135deg, #1479e9, #06418c);
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 42px;
    border: 1px solid #398eea;
    flex-shrink: 0;
}

.header-divider {
    width: 2px;
    height: 70px;
    background: #8db9e8;
    opacity: .7;
}

.header-title {
    font-size: clamp(22px, 3vw, 38px);
    font-weight: 800;
    color: white;
    margin: 0;
    line-height: 1.3;
}

.header-subtitle {
    color: #b8d2f5;
    font-size: 16px;
    margin-top: 7px;
}

.header-status {
    background: #102d53;
    border: 1px solid #2465a8;
    border-radius: 16px;
    padding: 15px 20px;
    min-width: 220px;
}

.status-active {
    color: #22df8a;
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 8px;
}

.model-info {
    color: #d4e4fa;
    font-size: 13px;
}

/* PANELS */

.panel {
    background: linear-gradient(145deg, #10213e, #0d1c36) !important;
    border: 1px solid #1b4c86 !important;
    border-radius: 22px !important;
    padding: 22px !important;
    box-shadow: 0 8px 25px rgba(0,0,0,.16);
}

.panel-heading {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 5px;
}

.heading-icon {
    width: 48px;
    height: 48px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1685f8, #0750b7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}

.panel-title {
    color: white;
    font-size: 23px;
    font-weight: 800;
    margin: 0;
}

.panel-description {
    color: #b4c9e7;
    font-size: 14px;
    margin: 8px 0 20px 62px;
}

/* INPUT */

textarea {
    background: #0a1931 !important;
    color: white !important;
    border: 1.5px solid #2788e8 !important;
    border-radius: 13px !important;
    font-size: 15px !important;
    min-height: 190px !important;
}

textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,.15) !important;
}

label {
    color: #e4efff !important;
    font-weight: 600 !important;
}

/* BUTTONS */

#analyze-btn button {
    background: linear-gradient(100deg, #1685f8, #0861d8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 54px;
    font-size: 16px !important;
    font-weight: 800 !important;
    box-shadow: 0 5px 15px rgba(22,133,248,.2);
    transition: .2s ease;
}

#analyze-btn button:hover {
    filter: brightness(1.12);
    transform: translateY(-2px);
}

#clear-btn button {
    background: #20395f !important;
    color: white !important;
    border: 1px solid #345783 !important;
    border-radius: 12px !important;
    min-height: 54px;
    font-size: 15px !important;
    font-weight: 700 !important;
}

#flag-btn button {
    background: linear-gradient(100deg, #ef4444, #c92727) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 54px;
    font-size: 15px !important;
    font-weight: 800 !important;
}

/* STATUS */

#status-message {
    color: #b8d2f5 !important;
    font-size: 13px;
    margin-top: 12px;
}

/* EXAMPLES */

.example-title {
    color: white;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 6px;
}

.example-description {
    color: #b4c9e7;
    font-size: 13px;
    margin-bottom: 15px;
}

.example-btn button {
    background: #142b4c !important;
    color: #e5f1ff !important;
    border: 1px solid #2367ad !important;
    border-radius: 11px !important;
    font-size: 12px !important;
    text-align: left !important;
    min-height: 48px;
}

.example-btn button:hover {
    background: #1b3e69 !important;
    border-color: #38bdf8 !important;
}

/* RESULTS */

#results-panel {
    min-height: 610px;
}

.result-empty {
    min-height: 460px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #9eb5d5;
    padding: 25px;
}

.empty-icon {
    font-size: 60px;
    margin-bottom: 20px;
}

.empty-title {
    color: white;
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 10px;
}

.empty-description {
    font-size: 13px;
    max-width: 300px;
    line-height: 1.8;
}

/* PREDICTION CARD */

.prediction-card {
    background: linear-gradient(110deg, #103c42, #103452);
    border: 2px solid #21d98b;
    border-radius: 18px;
    padding: 22px;
    margin: 12px 0 28px;
    display: grid;
    grid-template-columns: 90px 1fr 170px;
    align-items: center;
    gap: 18px;
}

.prediction-icon {
    width: 76px;
    height: 76px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1685f8, #0648a5);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 36px;
}

.prediction-main {
    border-left: 1px solid #4b718b;
    padding-left: 22px;
}

.prediction-label {
    color: #35e88b;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
}

.prediction-category {
    color: white;
    font-size: clamp(19px, 2vw, 28px);
    font-weight: 800;
    overflow-wrap: anywhere;
}

.confidence-box {
    border-left: 1px solid #4b718b;
    padding-left: 22px;
}

.confidence-label {
    color: #d4e5f8;
    font-size: 14px;
    margin-bottom: 7px;
}

.confidence-value {
    color: #22df8a;
    font-size: 29px;
    font-weight: 800;
}

/* PROBABILITIES */

.section-heading {
    color: white;
    font-size: 19px;
    font-weight: 800;
    margin: 20px 0;
}

.probability-row {
    margin-bottom: 17px;
}

.probability-info {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    color: #e0edff;
    font-size: 13px;
    margin-bottom: 8px;
}

.probability-name {
    overflow-wrap: anywhere;
}

.probability-value {
    font-weight: 800;
    white-space: nowrap;
}

.bar-background {
    width: 100%;
    height: 12px;
    background: #1c355a;
    border-radius: 20px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #1685f8, #38bdf8);
}

.bar-fill.predicted {
    background: linear-gradient(90deg, #1685f8, #22df8a);
}

.predicted-tag {
    color: #22df8a;
    font-size: 10px;
    font-weight: 800;
    margin-left: 5px;
}

/* INFORMATION */

.info-box {
    background: #122d54;
    border: 1px solid #1c5a9a;
    border-radius: 14px;
    padding: 17px;
    margin-top: 24px;
    color: #c7dcf8;
    font-size: 13px;
    line-height: 1.7;
}

.footer {
    background: linear-gradient(90deg, #0b2041, #071329);
    border-top: 1px solid #174b86;
    border-radius: 12px;
    color: #c6d9f3;
    text-align: center;
    padding: 22px;
    margin-top: 25px;
    font-size: 13px;
}

@media (max-width: 850px) {
    .gradio-container {
        padding: 10px !important;
    }

    #main-header {
        padding: 18px;
    }

    .header-left {
        gap: 10px;
    }

    .header-icon {
        width: 55px;
        height: 55px;
        font-size: 30px;
    }

    .header-divider {
        height: 55px;
    }

    .header-status {
        width: 100%;
    }

    .panel {
        padding: 16px !important;
    }

    .prediction-card {
        grid-template-columns: 1fr;
        text-align: center;
        justify-items: center;
    }

    .prediction-main,
    .confidence-box {
        border-left: none;
        padding-left: 0;
    }

    #results-panel {
        min-height: auto;
    }
}
"""


# =========================================================
# 4. EMPTY RESULTS
# =========================================================

EMPTY_STATE = """
<div class="result-empty">
    <div class="empty-icon">📊</div>
    <div class="empty-title">Prediction Results</div>
    <div class="empty-description">
        Your prediction results will appear here.
        Enter a banking complaint and click Submit
        to analyze it using our AI model.
    </div>
</div>
"""


# =========================================================
# 5. CATEGORY DISPLAY NAMES AND ICONS
# =========================================================

CATEGORY_NAMES = {
    "credit_card": "Credit Card",
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "mortgages_and_loans": "Mortgages and Loans",
    "retail_banking": "Retail Banking"
}

CATEGORY_ICONS = {
    "credit_card": "💳",
    "credit_reporting": "📄",
    "debt_collection": "📞",
    "mortgages_and_loans": "🏠",
    "retail_banking": "🏦"
}


def format_category(category):
    return CATEGORY_NAMES.get(
        category,
        category.replace("_", " ").title()
    )


# =========================================================
# 6. PREDICTION DISPLAY
# =========================================================

def display_prediction(text):
    if not text or not text.strip():
        return EMPTY_STATE, "⚠️ Please enter a complaint first."

    try:
        predictions = predict_complaint(text)

        if not predictions:
            return EMPTY_STATE, "⚠️ No prediction results returned."

        predicted_category = max(
            predictions,
            key=predictions.get
        )

        confidence = float(predictions[predicted_category]) * 100
        formatted_category = format_category(predicted_category)
        category_icon = CATEGORY_ICONS.get(predicted_category, "🏦")

        probability_html = ""

        sorted_predictions = sorted(
            predictions.items(),
            key=lambda item: item[1],
            reverse=True
        )

        for category, probability in sorted_predictions:
            percentage = max(
                0,
                min(100, float(probability) * 100)
            )

            is_predicted = category == predicted_category
            safe_category = html.escape(format_category(category))

            tag = (
                '<span class="predicted-tag">✓ PREDICTED</span>'
                if is_predicted else ""
            )

            bar_class = (
                "bar-fill predicted"
                if is_predicted else "bar-fill"
            )

            probability_html += f"""
            <div class="probability-row">
                <div class="probability-info">
                    <span class="probability-name">
                        {safe_category} {tag}
                    </span>
                    <span class="probability-value">
                        {percentage:.2f}%
                    </span>
                </div>
                <div class="bar-background">
                    <div class="{bar_class}"
                         style="width:{percentage:.2f}%;">
                    </div>
                </div>
            </div>
            """

        safe_formatted_category = html.escape(formatted_category)

        result_html = f"""
        <div class="prediction-card">
            <div class="prediction-icon">
                {category_icon}
            </div>
            <div class="prediction-main">
                <div class="prediction-label">
                    Predicted Category:
                </div>
                <div class="prediction-category">
                    {safe_formatted_category}
                </div>
            </div>
            <div class="confidence-box">
                <div class="confidence-label">
                    Confidence:
                </div>
                <div class="confidence-value">
                    {confidence:.2f}% ✓
                </div>
            </div>
        </div>

        <div class="section-heading">
            📊 Category Probabilities
        </div>

        {probability_html}

        <div class="info-box">
            ℹ️ The model predicts that this complaint
            belongs to the
            <strong>{safe_formatted_category}</strong>
            category, with a confidence score of
            <strong>{confidence:.2f}%</strong>.
        </div>
        """

        return result_html, "✅ Complaint analyzed successfully."

    except Exception as e:
        error_html = """
        <div class="result-empty">
            <div class="empty-icon">⚠️</div>
            <div class="empty-title">Prediction Error</div>
            <div class="empty-description">
                An error occurred while analyzing the complaint.
                Please try again.
            </div>
        </div>
        """
        print(f"Prediction error: {e}")
        return error_html, f"❌ Error: {str(e)}"


# =========================================================
# 7. CLEAR FUNCTION
# =========================================================

def clear_interface():
    return (
        "",
        EMPTY_STATE,
        "Ready to analyze a complaint."
    )


# =========================================================
# 8. FLAG BUTTON CALLBACK
# =========================================================

def flag_complaint(text):
    if not text or not text.strip():
        return "⚠️ Please enter a complaint before flagging."

    # UI feedback only. No permanent storage is configured.
    return (
        "🚩 Flag request received in the interface. "
        "Permanent storage has not been configured yet."
    )


# =========================================================
# 9. EXAMPLE COMPLAINTS
# =========================================================

example_texts = [
    "I noticed an unauthorized transaction on my credit card "
    "and I want the bank to investigate it.",

    "My credit report contains an incorrect late payment "
    "that does not belong to me.",

    "A debt collector keeps calling me about a debt "
    "that I do not recognize.",

    "My mortgage payment was processed incorrectly "
    "and the bank has not resolved the issue.",

    "My checking account was frozen without notification "
    "and I cannot access my money."
]


# =========================================================
# 10. GRADIO INTERFACE
# =========================================================

with gr.Blocks(
    css=custom_css,
    title="Bank Complaint Classification",
    theme=gr.themes.Base()
) as demo:

    # HEADER
    gr.HTML("""
    <div id="main-header">
        <div class="header-content">
            <div class="header-left">
                <div class="header-icon">🏦</div>
                <div class="header-divider"></div>
                <div>
                    <h1 class="header-title">
                        Bank Complaint Classification
                    </h1>
                    <div class="header-subtitle">
                        AI-Powered Customer Complaint Analysis
                    </div>
                </div>
            </div>

            <div class="header-status">
                <div class="status-active">
                    🟢 AI Model Active
                </div>
                <div class="model-info">
                    🤖 DistilBERT | 384 Tokens
                </div>
            </div>
        </div>
    </div>
    """)

    # MAIN CONTENT
    with gr.Row(equal_height=True):

        # LEFT COLUMN
        with gr.Column(scale=1):

            with gr.Column(elem_classes=["panel"]):

                gr.HTML("""
                <div class="panel-heading">
                    <div class="heading-icon">💬</div>
                    <div class="panel-title">
                        Customer Complaint
                    </div>
                </div>

                <div class="panel-description">
                    Describe your banking issue below:
                </div>
                """)

                complaint_input = gr.Textbox(
                    lines=8,
                    max_length=5000,
                    show_label=False,
                    placeholder="Describe your banking issue here...",
                    elem_id="complaint-input"
                )

                with gr.Row():
                    analyze_btn = gr.Button(
                        "➤  Submit",
                        variant="primary",
                        elem_id="analyze-btn",
                        scale=2
                    )

                    clear_btn = gr.Button(
                        "↻  Clear",
                        elem_id="clear-btn",
                        scale=1
                    )

                    flag_btn = gr.Button(
                        "🚩  Flag",
                        elem_id="flag-btn",
                        scale=1
                    )

                status_message = gr.Markdown(
                    "Ready to analyze a complaint.",
                    elem_id="status-message"
                )

            # EXAMPLES PANEL
            with gr.Column(elem_classes=["panel"]):

                gr.HTML("""
                <div class="example-title">
                    ⭐ Example Complaints
                </div>

                <div class="example-description">
                    Click an example to fill the input:
                </div>
                """)

                with gr.Row():
                    example_1 = gr.Button(
                        "Incorrect credit card charge",
                        elem_classes=["example-btn"]
                    )

                    example_2 = gr.Button(
                        "Inaccurate credit report",
                        elem_classes=["example-btn"]
                    )

                    example_3 = gr.Button(
                        "Disputed debt collection",
                        elem_classes=["example-btn"]
                    )

                with gr.Row():
                    example_4 = gr.Button(
                        "Incorrect mortgage payment",
                        elem_classes=["example-btn"]
                    )

                    example_5 = gr.Button(
                        "Retail banking problem",
                        elem_classes=["example-btn"]
                    )

        # RIGHT COLUMN
        with gr.Column(scale=1):
            with gr.Column(
                elem_classes=["panel"],
                elem_id="results-panel"
            ):
                gr.HTML("""
                <div class="panel-heading">
                    <div class="heading-icon">📊</div>
                    <div class="panel-title">
                        Prediction Results
                    </div>
                </div>
                """)

                results_output = gr.HTML(value=EMPTY_STATE)

    # FOOTER
    gr.HTML("""
    <div class="footer">
        🛡️ Secure &nbsp; • &nbsp;
        Reliable &nbsp; • &nbsp;
        AI-Powered
        <br><br>
        Bank Complaint Classification System
        <br>
        Developed by Hassan Saleh
    </div>
    """)

    # EVENTS
    analyze_btn.click(
        fn=display_prediction,
        inputs=complaint_input,
        outputs=[results_output, status_message],
        show_progress="full"
    )

    clear_btn.click(
        fn=clear_interface,
        inputs=None,
        outputs=[
            complaint_input,
            results_output,
            status_message
        ]
    )

    flag_btn.click(
        fn=flag_complaint,
        inputs=complaint_input,
        outputs=status_message
    )

    # EXAMPLE BUTTONS
    example_1.click(
        lambda: example_texts[0],
        outputs=complaint_input
    )

    example_2.click(
        lambda: example_texts[1],
        outputs=complaint_input
    )

    example_3.click(
        lambda: example_texts[2],
        outputs=complaint_input
    )

    example_4.click(
        lambda: example_texts[3],
        outputs=complaint_input
    )

    example_5.click(
        lambda: example_texts[4],
        outputs=complaint_input
    )


# =========================================================
# 11. LAUNCH
# =========================================================

if __name__ == "__main__":
    demo.queue().launch()
