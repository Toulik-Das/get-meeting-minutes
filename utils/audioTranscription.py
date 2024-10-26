import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_openai_api_key(api_key):
    """Set the OpenAI API key."""
    # openai.api_key = api_key

def transcribe_audio(audio_buffer, api_key):
    """Transcribe audio file to text using OpenAI's Whisper model.

    Args:
        audio_buffer (audio buffer): audio file buffer to transcribe.

    Returns:
        Optional[str]: The transcription text if successful, None otherwise.
    """
    try:
        logging.info(f"Starting transcription for file: {audio_buffer}")
        openai = OpenAI(api_key=api_key)

        # Open the audio file for transcription
        transcription = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_buffer,  # Use the in-memory buffer directly
            response_format="text"
            )
        
        logging.info("Transcription complete.")
        
        return transcription
    
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    
    return None

def audio_transcription(audio_buffer, api_key):
    """Main function to handle transcription."""

    if api_key:
        openai_config = set_openai_api_key(api_key)
        transcription = transcribe_audio(audio_buffer, api_key)
        
        if transcription:
            print("Transcription Result:\n", transcription)
        else:
            print("Transcription failed.")
        
        return transcription
    else:
        logging.error("API key is not set. Please set the OPENAI_API_KEY environment variable.")