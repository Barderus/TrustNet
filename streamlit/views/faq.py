import streamlit as st


def render():
    st.markdown("""
    <div class="hero">
        <div class="hero-title">FAQ</div>
        <div class="hero-subtitle">
            Frequently asked questions about the capstone, its models, and its limits.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h3>General Questions</h3></div>", unsafe_allow_html=True)

    with st.expander("What is TrustNet?"):
        st.write("""
        TrustNet is an undergraduate capstone that studies fake-news classification
        and headline/body stance detection. Its predictions reflect patterns in
        labeled datasets; they do not verify facts.
        """)

    with st.expander("How does TrustNet detect fake news?"):
        st.write("""
        The app loads a DistilBERT fake-news classifier for article text. A separate
        DistilBERT model predicts how a headline relates to an article body.
        The displayed token attributions help inspect a prediction, but they
        do not establish whether the content is true.
        """)

    with st.expander("What datasets were used?"):
        st.write("""
        The fake-news training workflow starts with Kaggle True/Fake News data
        and can include More Fake News data when available. The stance workflow
        uses FNC-1. FakeNewsNet titles are used in separate transfer and
        in-domain evaluations.
        """)

    st.markdown("<div class='card'><h3>Technical Questions</h3></div>", unsafe_allow_html=True)

    with st.expander("What ML models does TrustNet use?"):
        st.write("""
        The app uses two DistilBERT classifiers. The research comparison also
        includes TF-IDF linear baselines, TextCNN, and Bidirectional LSTM models.
        The app's token attribution uses transformers-interpret.
        """)

    with st.expander("How accurate is TrustNet?"):
        st.write("""
        Performance depends strongly on the dataset and input format. The
        Kaggle-trained model performed poorly on FakeNewsNet titles without
        retraining. See the Results Summary for saved scores and limitations.
        """)

    with st.expander("Can the predictions be explained?"):
        st.write("""
        The app shows token attributions for its transformer predictions. These
        are diagnostic clues about model behavior, not factual evidence.
        """)

    st.markdown("<div class='card'><h3>Privacy and Usage</h3></div>", unsafe_allow_html=True)

    with st.expander("Do you store my text?"):
        st.write("""
        The app code does not save submitted text to project files. Text is
        processed during the session to make a prediction and explanation.
        """)

    with st.expander("Can I use TrustNet for research?"):
        st.write("""
        TrustNet can be used to explore model behavior in a research or teaching
        setting. Its output should not be used as a factual verdict.
        """)

    with st.expander("How can I contact support?"):
        st.write("The Contact page is a demo form and does not send messages.")

    st.markdown("</div>", unsafe_allow_html=True)
