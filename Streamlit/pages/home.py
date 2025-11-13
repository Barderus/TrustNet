import streamlit as st

def render():
    st.title("TrustNet: Fake News Detector")

    st.markdown("""
    ### Project Summary  
    This is the landing page for TrustNet.  
    A fake news detection system built with Machine Learning and NLP.

    **This version shows only the UI layout** (no backend yet).
    """)

    st.markdown("---")
    st.subheader("Choose Input Method")

    tab_text, tab_file = st.tabs(["Paste Text", "Upload File"])

    with tab_text:
        st.markdown("#### Paste your text here:")
        st.text_area("Text Input", height=200)

    with tab_file:
        st.markdown("#### Upload a document:")
        st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

    st.markdown("---")

    st.button("Detect Fake News")

    st.markdown("### Results")
    st.info("Results will appear here once the model is connected.")
