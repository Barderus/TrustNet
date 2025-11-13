import streamlit as st

from views import home, about, contact

st.set_page_config(
    page_title="TrustNet",
    layout="wide"
)

PAGES = {
    "Home": home.render,
    "About Us": about.render,
    "Contact": contact.render,
}

def main():
    tab1, tab2, tab3 = st.tabs(["Home", "About Us", "Contact"])

    with tab1:
        home.render()
    with tab2:
        about.render()
    with tab3:
        contact.render()


if __name__ == "__main__":
    main()
