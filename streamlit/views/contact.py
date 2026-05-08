import streamlit as st

def render():

    # -------------------------
    # HERO SECTION
    # -------------------------
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Contact</div>
        <div class="hero-subtitle">
            Any feedback? Concern? Problems?
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # CONTACT CARD TITLE
    # -------------------------
    st.markdown("""
    <div class="card">
        <h3>Contact</h3>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # CONTACT FORM
    # -------------------------
    st.markdown(
        """
        <div style="text-align: center; max-width: 700px; margin: 0 auto; font-size: 18px; line-height: 1.5;">
            If you have questions, feedback, or would like to discuss this project,
            feel free to reach out using the form below.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Form container
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        subject = st.text_input("Subject")
        message = st.text_area("Message", height=150)

        submitted = st.form_submit_button("Send")

        if submitted:
            if not name or not email or not message:
                st.error("Please fill out name, email, and message before submitting.")
            else:
                st.success("Thanks for reaching out! Your message has been received.")
