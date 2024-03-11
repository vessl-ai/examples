import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the layout for the Streamlit app
st.set_page_config(layout="wide")


# Load the model and tokenizer
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", device_map="auto")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")

    return tokenizer, model

tokenizer, model = load_model()

# Settings section
with st.sidebar:
    st.title("Settings")
    token_length = st.slider("Token Length", 0, 512, 256)
    temperature = st.slider("Temperature", 0.0, 3.0, 1.0)
    top_p = st.slider("Top P", 0.0, 1.0, 0.5)

# Chat window setup
st.title("Gemma Chatbot")

user_input = st.text_input("Type your message here and press enter", key="input")
if user_input:
    # Convert input text to tokens
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    # Set the response generation configuration for the model
    chat_settings = {
        'max_length': token_length,
        'temperature': temperature,
        'top_p': top_p
    }
    output_ids = model.generate(input_ids, **chat_settings)

    # Convert generated tokens to string and add to response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    st.text(response)
