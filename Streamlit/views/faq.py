import streamlit as st

def render():
    # -------------------------
    # HERO SECTION
    # -------------------------
    st.markdown("""
    <div class="hero">
        <div class="hero-title">FAQ</div>
        <div class="hero-subtitle">
            Frequently Asked Questions about TrustNet - our mission, our models, and how the platform works.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Start main centered container
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # -------------------------
    # GENERAL QUESTIONS CARD
    # -------------------------
    st.markdown("""
    <div class="card">
        <h3>General Question </h3>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("What is TrustNet?"):
        st.write("""
        TrustNet is a machine-learning powered platform designed to detect 
        misleading or potentially false online news content using NLP models.
        """)

    with st.expander("How does TrustNet detect fake news?"):
        st.write("""
        TrustNet uses a multi-stage pipeline:
        - NLP preprocessing  
        - DistilBERT-based stance detection  
        - LSTM/BiLSTM classifiers  
        - SHAP visual explainability  

        This ensures accuracy, transparency, and reliable predictions.
        """)

    with st.expander("What datasets were used?"):
        st.write("""
        TrustNet is trained on several curated datasets:
        - FakeNewsNet  
        - Kaggle True/Fake News  
        - FNC-1 (Fake News Challenge)  

        These datasets provide strong context across multiple domains.
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # TECHNICAL QUESTIONS CARD
    # -------------------------
    st.markdown("""
    <div class="card">
        <h3>Technical Question </h3>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("What ML models does TrustNet use?"):
        st.write("""
        TrustNet uses:
        - DistilBERT for stance detection  
        - LSTM and BiLSTM sequence models  
        - Random Forest / XGBoost baselines  
        - SHAP for interpretability  
        """)

    with st.expander("How accurate is TrustNet?"):
        st.write("""
        Across multiple datasets:
        - Precision: ~0.92–0.96  
        - Recall: ~0.95  
        - F1 Score: ~0.94  

        Performance varies depending on domain and input style.
        """)

    with st.expander("Can the predictions be explained?"):
        st.write("""
        Yes! TrustNet includes SHAP explainability tools  
        that highlight the most influential words or patterns 
        contributing to the model's prediction.
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # PRIVACY CARD
    # -------------------------
    st.markdown("""
    <div class="card">
        <h3> Privacy & Usage </h3>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Do you store my text?"):
        st.write("""
        No.  
        TrustNet does **not** log, save, or store user-submitted text.  
        All processing occurs in-memory during your session only.
        """)

    with st.expander("Can I use TrustNet for research?"):
        st.write("""
        Yes! TrustNet is suitable for academic, journalism,
        and research workflows related to misinformation studies.
        """)

    with st.expander("How can I contact support?"):
        st.write("""
        You can reach out through the **Contact** page in the navigation bar.
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # Close main container
    st.markdown("</div>", unsafe_allow_html=True)
