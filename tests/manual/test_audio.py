import os
import streamlit as st
import streamlit.components.v1 as components
import base64

# Declare the component
_component_func = components.declare_component(
    "st_audiorec",
    path=os.path.join(
        os.path.dirname(__file__),
        "streamlit_audio_recorder",
        "st_audiorec",
        "frontend",
        "build"
    )
)

def st_audiorec():
    return _component_func()

# Set page config
st.set_page_config(page_title="Audio Recorder Test")

# Add some styling
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
            unsafe_allow_html=True)
st.markdown('''<style>.stAudio {height: 45px;}</style>''',
            unsafe_allow_html=True)

# Main app
st.title("Audio Recorder Test")

# Get audio data
wav_audio_data = st_audiorec()

# Display audio if available
if wav_audio_data is not None:
    st.write("Audio recorded! Playing back...")
    st.audio(wav_audio_data, format='audio/wav') 