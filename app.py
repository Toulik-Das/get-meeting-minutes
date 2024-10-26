import os
import io
import logging
import streamlit as st
from utils.extractAudio import extract_audio_from_video
from utils.audioTranscription import audio_transcription
from utils.lamaModelHandler import meeting_minutes_llm

LLAMA = st.secrets["LLAMA"]
HF_TOKEN = st.secrets["HF_TOKEN"]

# Configure logging
logging.basicConfig(level=logging.INFO)

def process_file_transcription(uploaded_file, api_key):
    """Process the uploaded file to generate transcripton of the meeting"""
    
    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
    audio_file = None

    # Create an in-memory file-like object for the audio extraction
    audio_buffer = io.BytesIO()

    # Determine file type and process accordingly
    try:
        if file_extension in [".mp4", ".mov"]:  # Check if the uploaded file is a video
            # Extract audio and write to an in-memory buffer
            audio_buffer = extract_audio_from_video(uploaded_file, audio_buffer)
            logging.info("Extracted audio from video.")
            audio_buffer.seek(0)  # Move the cursor to the beginning of the buffer
        
        elif file_extension in [".wav", ".mp3", ".m4a"]:  # Check if the uploaded file is audio
            audio_buffer = uploaded_file
            logging.info("Using uploaded audio file directly.")
        
        else:
            st.error("Unsupported file type. Please upload an audio or video file.")
            return None
        
        # Transcribe the audio to text
        transcription = audio_transcription(audio_buffer, api_key)
        logging.info("Transcription completed.")
        
        return transcription

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        st.error("An error occurred while processing the file. Please try again.")
        return None

# App Title and Logo
st.set_page_config(page_title="Meeting Minutes LLM", page_icon="📝", layout="wide")
st.title("📋 Meeting Minute")
st.write("Skip the note-taking and enjoy the conversation—just upload your meeting audio or video, and let us handle the minutes effortlessly!")

# Sidebar for API Key and LLM Selection
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Configure your GenAI settings here.")

    # API Key Input
    api_key = st.text_input("🔑 API Key", type="password", help="Enter your LLM API Key")

    # Choice of LLMs
    llm_choice = st.selectbox(
        "🤖 Choose your LLM",
        ["OpenAI GPT-4", "Anthropic Claude", "Cohere Command", "Custom Model"],
        help="Select the language model you want to use."
    )
    
    # Information about LLMs
    st.write("Note: Please ensure your API key matches the selected model for seamless integration.")

# Main Section for Audio/Video Upload
st.subheader("📁 Upload Meeting Audio/Video")
uploaded_file = st.file_uploader("Upload an audio or video file", type=["wav", "mp3", "m4a", "mp4", "mov"])

if uploaded_file and api_key:
    st.write("Generating minutes... please wait 🚀")

    transcription = process_file_transcription(uploaded_file, api_key)

    with st.spinner("Processing..."):
        meeting_minutes_generator = meeting_minutes_llm(transcription, LLAMA, HF_TOKEN)
        
        # Display each piece of generated text as it comes in
        for minutes in meeting_minutes_generator:
            st.text_area("📜 **Meeting Minutes Generated:**", value=minutes, height=300, key=str(minutes))

else:
    st.info("Please upload an audio or video file and provide your API key to start processing.")

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .css-1aumxhk, .css-18e3th9 {
        background: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
    }
    .st-bd {
        border: 2px solid #e1e5e9;
    }
    .stSidebar, .stButton>button {
        background: #0073e6 !important;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)