# Force sys.path to include the real project root
import sys
PROJECT_ROOT = r"C:\Users\Owner\PycharmProjects\TrustNet"
sys.path.append(PROJECT_ROOT)

import streamlit as st

from utils.text_processing import extract_text_from_pdf, extract_text_from_docx
from utils.model_loader import load_fake_news_model, load_stance_model
from utils.prediction import predict_text, get_word_attributions, clean_special_tokens, merge_wordpiece_tokens, top_k_attributions


def render():

    # ---------------------------------------------
    # Initialize session state
    # ---------------------------------------------
    if "last_task" not in st.session_state:
        st.session_state.last_task = "Fake News Detection"

    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    if "headline" not in st.session_state:
        st.session_state.headline = ""

    # -------------------------
    # HERO SECTION
    # -------------------------
    st.markdown("""
    <div class="hero">
        <div class="hero-title">TrustNet</div>
        <div class="hero-subtitle">
            A modern NLP platform for Fake News Detection and Stance Detection.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # -------------------------
    # PROJECT SUMMARY
    # -------------------------
    st.markdown("""
    <div class="card"><h3>Project Summary</h3></div>
    """, unsafe_allow_html=True)

    st.markdown("""
        TrustNet is a research platform exploring misinformation detection 
        and stance classification using DistilBERT-based NLP models.
    """, unsafe_allow_html=True)

    # -------------------------
    # TASK SELECTION
    # -------------------------
    st.markdown("""
    <div class="card"><h3>Select Task</h3></div>
    """, unsafe_allow_html=True)

    task = st.radio(
        "Choose which model to use:",
        ["Fake News Detection", "Stance Detection"],
        key="task_radio"
    )

    # -----------------------------------------------------
    # CLEAR INPUT WHEN SWITCHING TASKS
    # -----------------------------------------------------
    if task != st.session_state.last_task:
        st.session_state.user_text = ""
        st.session_state.headline = ""
        st.session_state.last_task = task

    # -------------------------
    # INPUT SECTION
    # -------------------------
    st.markdown("""
    <div class="card"><h3>Choose your input method</h3></div>
    """, unsafe_allow_html=True)

    # If stance detection, it requires headline input
    if task == "Stance Detection":
        st.markdown("### Headline")
        st.session_state.headline = st.text_input(
            "Enter the headline:",
            value=st.session_state.headline,
            key="headline_input"
        )
        headline = st.session_state.headline

    else:
        headline = None

    tab_text, tab_file = st.tabs(["Paste Text", "Upload File"])

    user_file = None

    with tab_text:
        st.session_state.user_text = st.text_area(
            "Paste your text here",
            height=200,
            key="body_text_area",
            value=st.session_state.user_text
        )


    with tab_file:
        user_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

    # -------------------------
    # DETECT BUTTON
    # -------------------------
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        detect = st.button(f"Run {task}")

    # -------------------------
    # RESULTS CARD
    # -------------------------
    st.markdown("""
    <div class="card"><h3>Results</h3></div>
    """, unsafe_allow_html=True)

    if detect:
        # For stance detection, headline required
        if task == "Stance Detection" and (not headline or headline.strip() == ""):
            st.error("Headline is required for stance detection.")
            st.stop()

        # Extract file text if uploaded
        user_text = st.session_state.user_text
        if user_file:
            if user_file.type == "application/pdf":
                user_text = extract_text_from_pdf(user_file)
            else:
                user_text = extract_text_from_docx(user_file)

        if not user_text.strip():
            st.error("Please enter or upload text.")
            st.stop()

        # Load proper model
        if task == "Fake News Detection":
            model, tokenizer = load_fake_news_model()
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            model_input = user_text

        else:  # stance detection
            model, tokenizer = load_stance_model()
            labels = ["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"]
            model_input = headline + " [SEP] " + user_text

        # Run prediction
        pred, probs, explain_text = predict_text(
            model,
            tokenizer,
            model_input,
            temperature=1.0
        )


        # Build probability rows
        rows_html = ""
        for lbl, p in zip(labels, probs):
            rows_html += f"<tr><td>{lbl}</td><td>{p:.4f}</td></tr>"

        # --- WHITE RESULT CARD ---
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
            unsafe_allow_html=True
        )

        # -------------------------
        # Compute Attributions
        # -------------------------
        word_attributions = get_word_attributions(model, tokenizer, explain_text)

        # Remove CLS / SEP
        word_attributions = clean_special_tokens(word_attributions)

        # Merge WordPiece tokens
        word_attributions = merge_wordpiece_tokens(word_attributions)

        # Get top 20
        word_attributions = top_k_attributions(word_attributions, k=20)

        # -------------------------
        # Build HTML rows
        # -------------------------
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

        # -------------------------
        # FULL HTML BLOCK (table)
        # -------------------------
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

        # Render via HTML (so Streamlit does NOT escape it)
        st.components.v1.html(html_block, height=600, scrolling=True)










