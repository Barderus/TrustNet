import streamlit as st

from pages import home, about, contact

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
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[choice]()

if __name__ == "__main__":
    main()
