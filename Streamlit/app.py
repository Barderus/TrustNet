import streamlit as st
from views import home, about, contact, faq

st.set_page_config(
    page_title="TrustNet",
    layout="wide"
)

PAGES = {
    "Home": home.render,
    "FAQ": faq.render,
    "About": about.render,
    "Contact": contact.render,
}

def inject_css():
    st.markdown("""
    <style>
    /* -------------------------
       FONT
    -------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* -------------------------
       BACKGROUND (INDIGO / PURPLE + ABSTRACT)
    -------------------------- */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a; /* deep slate */
        background-image:
            radial-gradient(circle at 15% 20%, rgba(99,102,241,0.25) 0%, transparent 60%),
            radial-gradient(circle at 80% 80%, rgba(168,85,247,0.22) 0%, transparent 60%),
            linear-gradient(135deg, #1e1b4b 0%, #0f172a 60%, #0b1120 100%);
        background-attachment: fixed;
    }

    /* -------------------------
       SIDEBAR
    -------------------------- */
    section[data-testid="stSidebar"] {
        background-color: #f3f4ff;
        border-right: 1px solid #e0e7ff;
    }

    /* -------------------------
       TYPOGRAPHY — LIGHT MODE
    -------------------------- */
    h1, h2, h3, h4 {
        color: black !important;  /* light slate */
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    
    p, label, span, div, li {
        color: #e2e8f0 !important;  /* light gray */
    }
    
    /* Subtle secondary text */
    .subtle-text, small {
        color: #cbd5e1 !important;
    }


    /* -------------------------
       MAIN LAYOUT CONTAINER
       (use by wrapping content in <div class="main-container"> ... </div>)
    -------------------------- */
    .main-container {
        max-width: 960px;
        margin: 0 auto;
        padding: 1.5rem 1.5rem 3rem;
    }

    /* -------------------------
       TOP NAVIGATION BAR
    -------------------------- */
    .navbar {
        width: 100%;
        background: rgba(255, 255, 255, 0.75); /* semi-transparent */
        backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(203, 213, 225, 0.4);
        position: sticky;
        top: 0;
        z-index: 999;
        padding: 0.8rem 0;
    }
    
    .navbar-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 0 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .nav-left {
        font-size: 1.4rem;
        font-weight: 1000;
        color: #4f46e5 !important;
        letter-spacing: -0.02em;
    }
    
    .nav-right a {
        margin-left: 1.8rem;
        text-decoration: none;
        font-size: 1rem;
        font-weight: 500;
        color: #374151;
        transition: all 0.2s ease;
        padding-bottom: 4px;
        border-bottom: 2px solid transparent;
    }
    
    .nav-right a:hover {
        color: #111827;
        border-bottom: 2px solid #6366f1;
    }
    
    .nav-active {
        color: #111827 !important;
        border-bottom: 2px solid #6366f1 !important;
    }

    /* -------------------------
       CARDS (SEMI-TRANSPARENT)
    -------------------------- */
    /* Modern Elevated Card */
    .card {
        background: rgba(255, 255, 255, 0.07); /* transparent glass */
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);

    
        padding: 0.9rem 1.2rem;
        border-radius: 16px;
    
        border: 1px solid rgba(255, 255, 255, 0.12);
    
        box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.25),
            inset 0 0 0 1px rgba(255, 255, 255, 0.05);
    
        transition: transform 0.18s ease, box-shadow 0.2s ease;
        margin-bottom: 1.5rem;
    }
    
    /* Hover elevation */
    .card:hover {
        transform: translateY(-4px);
        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.35),
            inset 0 0 0 1px rgba(255, 255, 255, 0.08);
    }
    
    /* Card Title Beautification */
    .card h3 {
        color: #f1f5f9 !important;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0 !important;
        margin-top: 0 !important;
        letter-spacing: -0.01em;
    }



    /* -------------------------
       PRIMARY BUTTONS (PILL, RAISED)
    -------------------------- */
    div.stButton > button {
        background-color: #6366f1;
        color: #ffffff;
        padding: 0.8rem 1.8rem;
        border-radius: 999px;
        border: none;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.35);
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        background-color: #4f46e5;
        box-shadow: 0 14px 30px rgba(79, 70, 229, 0.45);
        transform: translateY(-1px);
    }

    /* -------------------------
       FILE UPLOADER BUTTON (MATCH PRIMARY)
    -------------------------- */
    [data-testid="stFileUploader"] button {
        background-color: #6366f1 !important;
        color: #ffffff !important;
        padding: 0.8rem 1.8rem !important;
        border-radius: 999px !important;
        border: none !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background-color: #4f46e5 !important;
        box-shadow: 0 14px 30px rgba(79, 70, 229, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    
    /* FILE UPLOADER CONTAINER */
    [data-testid="stFileUploader"] section {
        background: #ffffff;
        border-radius: 14px;
        padding: 0.9rem;
        border: 1px dashed #c7d2fe;
    }
    
    /* -------------------------
       FORCE WHITE TEXT ON ALL BUTTONS
    -------------------------- */
    div.stButton > button, 
    .stButton button, 
    button[kind="primary"], 
    button[kind="secondary"], 
    button {
        color: #ffffff !important;
    }


    /* -------------------------
       TEXT AREAS
    -------------------------- */
    textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-radius: 12px !important;
        border: 1px solid #cbd5f5 !important;
        padding: 0.75rem !important;
    }

    /* -------------------------
       TABS (BLUE UNDERLINE, MODERN)
    -------------------------- */
    /* Tab container spacing */
    .stTabs {
        margin-top: 0.5rem;
    }

    /* Individual tab buttons */
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: #6b7280;
        border-bottom: 2px solid transparent;
        padding: 0.5rem 1rem;
    }

    /* Active tab */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #111827;
        border-bottom: 2px solid #6366f1;
    }

    /* Hover state */
    .stTabs [data-baseweb="tab"]:hover {
        color: #111827;
        background-color: rgba(99, 102, 241, 0.06);
        border-radius: 999px 999px 0 0;
    }
    
    /* -------------------------
   HERO SECTION
    -------------------------- */
    
    .hero {
        text-align: center;
        padding: 6rem 1rem 4rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        color: #4b5563;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.7;
    }
    
    /* Hero call-to-action button */
    .hero-cta {
        margin-top: 2.5rem;
    }
    
    .hero-cta button {
        background-color: #6366f1 !important;
        color: white !important;
        padding: 0.9rem 2rem !important;
        border-radius: 999px !important;
        border: none !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 14px 30px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    
    .hero-cta button:hover {
        background-color: #4f46e5 !important;
        box-shadow: 0 18px 38px rgba(79, 70, 229, 0.45) !important;
        transform: translateY(-2px) !important;
    }

/* -------------------------
   FIX: TAB TEXT COLOR (ALWAYS DARK)
-------------------------- */

/* Base tab text */
.stTabs [data-baseweb="tab"] {
    color: #374151 !important;   /* Slate gray */
    font-weight: 500;
}

/* Active tab text */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #111827 !important;   /* Near-black */
}

/* Hover state text */
.stTabs [data-baseweb="tab"]:hover {
    color: #111827 !important;
}


    /* -------------------------
       HEADER / FOOTER
    -------------------------- */
    [data-testid="stHeader"] {
        background: transparent;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        padding: 2rem 0 0;
    }
    
    /* FORCE WHITE TEXT ON ALL BUTTONS + INNER ELEMENTS */
    div.stButton > button, 
    div.stButton > button * ,
    button, 
    button * ,
    .st-emotion-cache-19rxjzo, 
    .st-emotion-cache-19rxjzo * ,
    [data-testid="baseButton-primary"], 
    [data-testid="baseButton-primary"] *, 
    [data-testid="baseButton-secondary"], 
    [data-testid="baseButton-secondary"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    </style>
    """, unsafe_allow_html=True)



def render_navbar(active_page):
    st.markdown(f"""
<div class="navbar">
    <div class="navbar-container">
        <div class="nav-left">TRUSTNET</div>
        <div class="nav-right">
            <a href="/?nav=Home" class="{ 'nav-active' if active_page=='Home' else '' }" target="_self">Home</a>
            <a href="/?nav=FAQ" class="{ 'nav-active' if active_page=='FAQ' else '' }" target="_self">FAQ</a>
            <a href="/?nav=About" class="{ 'nav-active' if active_page=='About' else '' }" target="_self">About</a>
            <a href="/?nav=Contact" class="{ 'nav-active' if active_page=='Contact' else '' }" target="_self">Contact</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)




def main():
    inject_css()

    # -- NEW WAY: use st.query_params --
    params = st.query_params
    active_page = params.get("nav", "Home")

    # Render navbar with active highlight
    render_navbar(active_page)

    # Render the selected page
    PAGES[active_page]()

    # Minimal footer
    st.markdown('<div class="footer">TrustNet © 2025</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
