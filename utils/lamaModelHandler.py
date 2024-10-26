import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
from transformers import TextStreamer
from huggingface_hub import login 

def set_hugging_face_api_key(api_key):
    """Set the Hugging Face API key."""
    login(api_key)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_model_and_tokenizer(model_name):
    """Load the tokenizer and model."""
    try:
        # Set up the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Set up the model with quantization configuration
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config)
        
        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        raise

def generate_meeting_minutes(transcription, tokenizer, model):
    """Generate meeting minutes from transcription."""
    system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways, and action items with owners, in markdown."
    user_prompt = f"Below is an extract transcript of a Enterprise Data Cataloging and Marketplace teams meeting. Please write minutes in markdown, including a summary with attendees, location, and date; discussion points; takeaways; and action items with owners.\n{transcription}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        streamer = TextStreamer(tokenizer)
        outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

        #Stream Output
        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            yield response  # Yield the intermediate output

        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return response

    except Exception as e:
        logging.error(f"Error during generation: {e}")
        raise

def meeting_minutes_llm(transcription, LLAMA_MODEL, api_key):
    """Main function to generate minutes from transcription."""
    logging.info("Starting the meeting minutes generation process.")

    # Hugging Face API Key
    set_hugging_face_api_key(api_key)
    
    # Load model and tokenizer
    tokenizer, model = setup_model_and_tokenizer(LLAMA_MODEL)

    # Generate minutes
    for mom in generate_meeting_minutes(transcription, tokenizer, model):
        yield mom  # Yield each portion of generated text