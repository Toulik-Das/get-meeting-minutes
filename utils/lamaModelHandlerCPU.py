import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login

def set_hugging_face_api_key(api_key):
    """Set the Hugging Face API key."""
    login(api_key)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_pipeline(model_name):
    """Load the text generation pipeline."""
    try:
        # Set up the pipeline with quantization if CUDA is available
        if torch.cuda.is_available():
            logging.info("USING CUDA")
            from bitsandbytes import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            # Load the model and tokenizer into the pipeline
            gen_pipeline = pipeline("text-generation", model=model_name, device=0, quantization_config=quant_config)
        else:
            logging.info("USING CPU")
            gen_pipeline = pipeline("text-generation", model=model_name)

        return gen_pipeline
    except Exception as e:
        logging.error(f"Error loading model into pipeline: {e}")
        raise

def generate_meeting_minutes(transcription, pipeline):
    """Generate meeting minutes from transcription."""
    system_message = (
        "You are an assistant that produces minutes of meetings from transcripts, "
        "with summary, key discussion points, takeaways, and action items with owners, in markdown."
    )

    user_prompt = f"Below is an extract transcript of an Enterprise Data Cataloging and Marketplace teams meeting. Please write minutes in markdown, including a summary with attendees, location, and date; discussion points; takeaways; and action items with owners.\n{transcription}"

    messages = [system_message, user_prompt]
    
    try:
        # Generate output using the pipeline
        response = pipeline(messages, max_length=2000, num_return_sequences=1)[0]['generated_text']
        return response

    except Exception as e:
        logging.error(f"Error during generation: {e}")
        raise

def meeting_minutes_llm(transcription, LLAMA_MODEL, api_key):
    """Main function to generate minutes from transcription."""
    logging.info("Starting the meeting minutes generation process.")

    # Hugging Face API Key
    set_hugging_face_api_key(api_key)

    # Load the generation pipeline
    gen_pipeline = setup_pipeline(LLAMA_MODEL)

    # Generate minutes
    minutes = generate_meeting_minutes(transcription, gen_pipeline)
    yield minutes