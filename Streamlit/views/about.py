import streamlit as st

def render():
    st.title("About TrustNet")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    TrustNet is a research project dedicated to improving misinformation detection
    through modern Machine Learning and Natural Language Processing methods.

    The system is designed to explore model performance, interpretability, and
    real-world applicability across various datasets and text sources.

    This page is currently a placeholder and will be expanded with additional
    project details, methodology, and research context.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
