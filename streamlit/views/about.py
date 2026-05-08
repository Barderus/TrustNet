import streamlit as st

def render():
    # -------------------------
    # HERO SECTION
    # -------------------------
    st.markdown("""
    <div class="hero">
        <div class="hero-title">About</div>
        <div class="hero-subtitle">
            Everything to know about TrustNet and its developer
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>TrustNet </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
            TrustNet (FakeNews Finder) is an interactive web tool that helps users check whether a news article or text appears credible. 
            Users can paste an article or link, and the system produces a trustworthiness score (0–100) along with an explanation of 
            why the content was judged that way. The goal is not only to detect misinformation but to teach critical reading by showing which words, 
            tones, and claims influenced the decision.
            
            This project was developed as part of my undergraduate Capstone in Computer Science. 
            It combines fake-news classification, stance detection, and explainability to make AI-based fact-checking more transparent.

        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">
            <h3>Developer</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("""
    I’m Gabriel dos Reis, a Computer Science student passionate about machine learning, NLP, and model interpretability. 
    My work focuses on building transparent AI systems that go beyond classification to explain their reasoning. 
    For TrustNet, I combined fake-news classification, stance detection, and SHAP-based explainability to explore real-world misinformation challenges. 
    I enjoy taking on complex datasets, experimenting with model architectures, and translating research concepts into practical, 
    interactive applications.

    """)
    st.markdown("</div>", unsafe_allow_html=True)
