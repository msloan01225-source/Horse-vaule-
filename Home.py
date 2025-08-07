import streamlit as st

# ---- Page Config ----
st.set_page_config(page_title="🏠 BetEdge Home", layout="wide")

# ---- Background Styling ----
page_bg = '''
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3, h4, h5, h6, .stMarkdown {
    color: white !important;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
'''
st.markdown(page_bg, unsafe_allow_html=True)

# ---- Title & Video Section ----
st.markdown("<h1 style='text-align: center;'>Welcome to BetEdge 🧠</h1>", unsafe_allow_html=True)

# Placeholder video link – replace with your actual hosted video
st.video("https://user-images.githubusercontent.com/00000000/betedge_intro_demo.mp4")

# ---- Intro Section ----
st.markdown("### 🔍 What is BetEdge?")
st.markdown(
    "**Smarter Betting. Real Value. Every Time.**  \n"
    "We combine AI, real-time bookie data, and predictive insights to bring you "
    "**market-beating betting intelligence.**"
)

st.markdown("---")

# ---- Feature Highlights ----
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🧠 EdgeBrain™")
    st.write("Your AI-powered betting assistant. Constantly scanning, learning and calculating.")

with col2:
    st.subheader("📊 Market Scanning")
    st.write("We fetch and filter odds, value %, and live market movement from top bookies.")

with col3:
    st.subheader("🏆 Value Leaderboards")
    st.write("See who’s top of the charts today for Win, Place, and Edge %.")

st.markdown("---")
st.markdown("### 🚀 Where do you want to go?")

# ---- Navigation Cards ----
nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    if st.button("🐎 Horse Racing"):
        st.switch_page("pages/1_Horse_Racing.py")

with nav2:
    if st.button("⚽ Football (Beta)"):
        st.switch_page("pages/2_Football.py")

with nav3:
    if st.button("🧠 EdgeBrain Picks"):
        st.switch_page("pages/3_EdgeBrain.py")

with nav4:
    if st.button("🎓 How It Works"):
        st.switch_page("pages/4_How_It_Works.py")

# ---- Footer ----
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2025 BetEdge. Smarter Bets, Better Returns.</p>", unsafe_allow_html=True)
