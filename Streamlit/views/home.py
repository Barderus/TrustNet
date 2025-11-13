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
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ### Project Summary
    TrustNet is a research-driven fake news detection system that leverages
    advanced NLP and machine learning models to identify misleading or false information.
    This UI demonstrates the interface layout before model integration.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # INPUT CARD
    # -------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Choose Input Method")

    tab_text, tab_file = st.tabs(["Paste Text", "Upload File"])

    with tab_text:
        st.markdown("#### Paste your text here:")
        st.text_area("Text Input", height=200)

    with tab_file:
        st.markdown("#### Upload a document:")
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
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Results")
    st.info("Results will appear here once the model is connected.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Close main container
    st.markdown("</div>", unsafe_allow_html=True)
