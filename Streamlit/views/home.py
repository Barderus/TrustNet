# Force sys.path to include the real project root
import sys
PROJECT_ROOT = r"C:\Users\Owner\PycharmProjects\TrustNet"
sys.path.append(PROJECT_ROOT)

import streamlit as st

from utils.text_processing import extract_text_from_pdf, extract_text_from_docx
from utils.model_loader import load_fake_news_model, load_stance_model
from utils.prediction import run_prediction


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
        headline = st.text_input(
            "Enter the headline:",
            value=st.session_state.headline,
            key="headline_input"
        )
    else:
        headline = None

    tab_text, tab_file = st.tabs(["Paste Text", "Upload File"])

    user_file = None

    with tab_text:
        text_value = st.text_area(
            "Paste your text here",
            height=200,
            value=st.session_state.user_text,
            key="body_text_area"
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
        user_text = text_value
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
            labels = ["FAKE", "REAL"]
            model_input = user_text

        else:  # stance detection
            model, tokenizer = load_stance_model()
            labels = ["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"]

            # Combine headline + body (stance models expect this format)
            model_input = headline + " [SEP] " + user_text

        # Run prediction
        pred, probs = run_prediction(model, tokenizer, model_input)

        st.success(f"Prediction: **{labels[pred]}**")

        st.subheader("Class Probabilities:")
        for lbl, p in zip(labels, probs):
            st.write(f"{lbl}: **{p:.4f}**")

    else:
        st.info("Results will appear here once the model is connected.")

    st.markdown("</div>", unsafe_allow_html=True)
