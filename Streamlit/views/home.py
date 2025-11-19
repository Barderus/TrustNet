import streamlit as st


def render():
    # -------------------------
    # HERO SECTION
    # -------------------------
    st.markdown("""
    <div class="hero">
        <div class="hero-title">TrustNet</div>
        <div class="hero-subtitle">
            A modern fake news detection platform powered by Machine Learning 
            and Natural Language Processing. Analyze text or documents and get 
            predictions backed by explainable AI.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Start main centered container
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # -------------------------
    # PROJECT SUMMARY CARD
    # -------------------------
    st.markdown("""
    <div class="card">
        <h3>Project Summary</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        TrustNet is a research project dedicated to improving misinformation detection through modern Machine Learning and Natural Language Processing methods.
            
        The system is designed to explore model performance, interpretability, and real-world applicability across various datasets and text sources.
        """,
        unsafe_allow_html=True
    )



    # -------------------------
    # INPUT CARD
    # -------------------------
    st.markdown("""
    <div class="card">
        <h3>Choose your input method </h3>
    </div>
    """, unsafe_allow_html=True)

    tab_text, tab_file = st.tabs(["Paste Text", "Upload File"])

    with tab_text:
        st.text_area("Paste your text here", height=200)


    with tab_file:
        st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])


    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # DETECT BUTTON (centered)
    # -------------------------
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Detect Fake News")

    # -------------------------
    # RESULTS CARD
    # -------------------------
    st.markdown("""
    <div class="card">
        <h3>Results </h3>
    </div>
    """, unsafe_allow_html=True)
    st.info("Results will appear here once the model is connected.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Close main container
    st.markdown("</div>", unsafe_allow_html=True)
