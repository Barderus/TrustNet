import streamlit as st


def render():
    st.markdown("""
    <div class="hero">
        <div class="hero-title">About</div>
        <div class="hero-subtitle">
            TrustNet and the questions behind the capstone
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'><h3>TrustNet</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    TrustNet is an undergraduate Computer Science capstone about fake-news
    classification and headline/body stance detection. The app uses separate
    DistilBERT models for these tasks and shows class probabilities and token
    attributions. A fake-news prediction describes similarity to labeled
    training examples; it does not check the facts in an article. Stance describes
    how a headline relates to a body, not whether either is true.
    """)

    st.markdown("<div class='card'><h3>Developer</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    I am Gabriel dos Reis. I built TrustNet to compare NLP models, study how
    their results change across datasets, and examine the mistakes they make.
    The FakeNewsNet title-only evaluation is one example of why a strong score
    on one dataset should be interpreted carefully.
    """)
