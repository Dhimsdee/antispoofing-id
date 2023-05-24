# main.py
import streamlit as st
from activenessnet import activenessnet
from passivenessnet import detect_passive_liveness
from serial import serial_detection

# Add the logo to the app header
st.image('./Logo full white.png', width=200)

# create a dropdown menu for each liveness detection type
liveness_type = st.selectbox(
    "Silakan pilih metode liveness detection yang diinginkan:", 
    ("Pilih salah satu metode di sini", "Aktif", "Pasif", "Serial")
)

if liveness_type == "Aktif":
    st.title("Liveness Detection Aktif")
    if st.button("Detect Liveness"):
        activenessnet()

elif liveness_type == "Pasif":
    st.title("Liveness Detection Pasif")
    if st.button("Detect Liveness"):
        detect_passive_liveness()

elif liveness_type == "Serial":
    st.title("Liveness Detection Serial")
    if st.button("Detect Liveness"):
        stframe = st.empty()
        # serial_detection()

        # Step 1: Passive Detection
        label = serial_detection()
        if label == "fake":
            st.error("Deteksi pasif gagal. Anda tidak dapat melanjutkan proses deteksi selanjutnya")
        else:
            # Step 2: Active Detection
            label = activenessnet()

        stframe.empty()

# Hide the Streamlit menu
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
