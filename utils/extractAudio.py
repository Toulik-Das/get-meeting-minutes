import os
import logging
from moviepy.editor import VideoFileClip
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure you set this environment variable

def extract_audio_from_video(mp4_file, mp3_file):
    try:
        duration_minutes=20
        
        # Check if the input file exists
        if not os.path.isfile(mp4_file):
            logging.error(f"Input video file not found: {mp4_file}")
            return None
        
        # Load the video clip
        logging.info(f"Loading video file: {mp4_file}")
        video_clip = VideoFileClip(mp4_file)
        
        # Set the duration for extraction in seconds
        duration = duration_minutes * 60
        
        # Ensure the duration does not exceed the video length
        if duration > video_clip.duration:
            logging.warning(f"Duration specified ({duration_minutes} minutes) exceeds video length. Adjusting to video length.")
            duration = video_clip.duration
        
        # Extract the audio from the video clip for the specified duration
        logging.info(f"Extracting audio for the first {duration_minutes} minutes...")
        audio_clip = video_clip.audio.subclip(0, duration)
        
        # Write the audio to a separate file
        audio_clip.write_audiofile(mp3_file)
        
        logging.info(f"Audio extraction successful! Saved to: {mp3_file}")
        
        return mp3_file  # Return the path of the saved audio file
    
    except Exception as e:
        logging.error(f"An error occurred during audio extraction: {e}")
        return None
    
    finally:
        # Close the video and audio clips if they exist
        try:
            audio_clip.close()
            video_clip.close()
        except:
            pass  # Ignore errors during cleanup
