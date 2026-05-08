from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from utils.model_loader import load_fake_news_model, load_stance_model
from utils.prediction import (
    clean_special_tokens,
    get_word_attributions,
    merge_wordpiece_tokens,
    predict_text,
    top_k_attributions,
)
from utils.text_processing import extract_text_from_docx, extract_text_from_pdf


def render():
    if "last_task" not in st.session_state:
        st.session_state.last_task = "Fake News Detection"

    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    if "headline" not in st.session_state:
        st.session_state.headline = ""

    st.markdown(
        """
    <div class="hero">
        <div class="hero-title">TrustNet</div>
        <div class="hero-subtitle">
            A modern NLP platform for Fake News Detection and Stance Detection.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.markdown(
        """
    <div class="card"><h3>Project Summary</h3></div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        TrustNet is a research platform exploring misinformation detection
        and stance classification using DistilBERT-based NLP models.
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="card"><h3>Select Task</h3></div>
    """,
        unsafe_allow_html=True,
    )

    task = st.radio(
        "Choose which model to use:",
        ["Fake News Detection", "Stance Detection"],
        key="task_radio",
    )

    if task != st.session_state.last_task:
        st.session_state.user_text = ""
        st.session_state.headline = ""
        st.session_state.last_task = task

    st.markdown(
        """
    <div class="card"><h3>Choose your input method</h3></div>
    """,
        unsafe_allow_html=True,
    )

    if task == "Stance Detection":
        st.markdown("### Headline")
        st.session_state.headline = st.text_input(
            "Enter the headline:",
            value=st.session_state.headline,
            key="headline_input",
        )
        headline = st.session_state.headline
    else:
        headline = None

    tab_text, tab_file = st.tabs(["Paste Text", "Upload File"])

    with tab_text:
        st.session_state.user_text = st.text_area(
            "Paste your text here",
            height=200,
            key="body_text_area",
            value=st.session_state.user_text,
        )

    with tab_file:
        user_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        detect = st.button(f"Run {task}")

    st.markdown(
        """
    <div class="card"><h3>Results</h3></div>
    """,
        unsafe_allow_html=True,
    )

    if detect:
        if task == "Stance Detection" and (not headline or headline.strip() == ""):
            st.error("Headline is required for stance detection.")
            st.stop()

        user_text = st.session_state.user_text
        if user_file:
            if user_file.type == "application/pdf":
                user_text = extract_text_from_pdf(user_file)
            else:
                user_text = extract_text_from_docx(user_file)

        if not user_text.strip():
            st.error("Please enter or upload text.")
            st.stop()

        try:
            if task == "Fake News Detection":
                model, tokenizer = load_fake_news_model()
                labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
                model_input = user_text
            else:
                model, tokenizer = load_stance_model()
                labels = ["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"]
                model_input = headline + " [SEP] " + user_text

            pred, probs, explain_text = predict_text(
                model,
                tokenizer,
                model_input,
                temperature=1.0,
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

        rows_html = ""
        for label, probability in zip(labels, probs):
            rows_html += f"<tr><td>{label}</td><td>{probability:.4f}</td></tr>"

        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-title">
                Prediction: <span class="prediction-value">{labels[pred]}</span>
              </div>

              <table class="result-table">
                <tr><th>Class</th><th>Probability</th></tr>
                {rows_html}
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        word_attributions = get_word_attributions(model, tokenizer, explain_text)
        word_attributions = clean_special_tokens(word_attributions)
        word_attributions = merge_wordpiece_tokens(word_attributions)
        word_attributions = top_k_attributions(word_attributions, k=20)

        rows = ""
        for token, score in word_attributions:
            color = "#22c55e" if score > 0 else "#ef4444" if score < 0 else "#374151"
            rows += f"""
                <tr>
                    <td style="padding: 6px 10px; border: 1px solid #e5e7eb;">{token}</td>
                    <td style="padding: 6px 10px; border: 1px solid #e5e7eb; color:{color};">
                        {score:.4f}
                    </td>
                </tr>
            """

        html_block = f"""
        <div class="result-card" style="background:white; padding: 1.5rem;">
            <h3 style="color:#111827; margin-bottom: 1em;"> Most Influential Tokens</h3>

            <table class="custom-table" style="width:100%; border-collapse: collapse; font-size: 0.95rem;">
                <tr style="background:#f3f4f6;">
                    <th style="padding: 8px; border: 1px solid #e5e7eb;">Token</th>
                    <th style="padding: 8px; border: 1px solid #e5e7eb;">Attribution Score</th>
                </tr>
                {rows}
            </table>
        </div>
        """

        st.components.v1.html(html_block, height=600, scrolling=True)
    else:
        st.info("Results will appear here once the model is connected.")

    st.markdown("</div>", unsafe_allow_html=True)
