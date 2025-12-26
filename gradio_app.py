"""
Health Chatbot Gradio Interface
Provides a web interface with voice input support for the health chatbot.
"""

import gradio as gr
from agent import get_chatbot_response
from openai import OpenAI

client = OpenAI()


def transcribe_audio(audio_path):
    """Transcribe audio file to text using Whisper API."""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Transcription error: {e}")
        return None


import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os

def save_audio_to_temp(audio_data):
    """Save numpy audio data to a temporary wav file."""
    if not audio_data:
        return None
    
    sample_rate, data = audio_data
    
    # Create temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp_audio.wav")
    
    # Save as WAV
    wavfile.write(temp_path, sample_rate, data)
    return temp_path


def process_input(text_input, audio_input, history):
    """
    Process both text and audio inputs.
    
    Args:
        text_input: Text from the textbox
        audio_input: Audio data (sample_rate, numpy_array) from microphone
        history: Chat history (list of dicts)
        
    Returns:
        Updated history, empty text input, None audio input
    """
    user_message = ""
    
    # Priority: text input first, then audio
    if text_input and text_input.strip():
        user_message = text_input
    elif audio_input:
        # Save numpy audio to temp file for Whisper
        audio_path = save_audio_to_temp(audio_input)
        if audio_path:
            transcribed = transcribe_audio(audio_path)
            # Cleanup temp file
            try:
                os.remove(audio_path)
            except:
                pass
                
            if transcribed:
                user_message = transcribed
            else:
                new_history = history + [{"role": "assistant", "content": "❌ Sorry, I couldn't transcribe the audio. Please try again."}]
                return new_history, "", None
        else:
            new_history = history + [{"role": "assistant", "content": "❌ Error processing audio data."}]
            return new_history, "", None
    
    if not user_message:
        new_history = history + [{"role": "assistant", "content": "Please provide a message via text or voice."}]
        return new_history, "", None
    
    # Add user message to history
    new_history = history + [{"role": "user", "content": user_message}]
    
    # Get chatbot response
    try:
        response = get_chatbot_response(user_message)
        new_history.append({"role": "assistant", "content": response})
    except Exception as e:
        print(f"Chat error: {e}")
        new_history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
    
    return new_history, "", None


# Build the interface with Blocks
with gr.Blocks(title="🏥 Health Chatbot") as demo:
    gr.Markdown("# 🏥 Health Chatbot")
    gr.Markdown(
        "Hi! I'm your health information assistant. Ask me about symptoms, "
        "conditions, or general health questions.\n\n"
        "💬 Type your question or 🎤 record your voice."
    )
    
    chatbot = gr.Chatbot(
        label="Chat",
        height=500
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            text_input = gr.Textbox(
                label="Type your message",
                placeholder="What can I help you with?",
                lines=2
            )
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",  # Changed to numpy to avoid IDM download triggering
                label="🎤 Record Voice"
            )
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")
    
    gr.Examples(
        examples=[
            "What are the symptoms of flu?",
            "How can I improve my sleep quality?",
            "What should I do if I have a headache?",
        ],
        inputs=text_input,
    )
    
    # Event handlers
    submit_btn.click(
        process_input,
        inputs=[text_input, audio_input, chatbot],
        outputs=[chatbot, text_input, audio_input]
    )
    
    text_input.submit(
        process_input,
        inputs=[text_input, audio_input, chatbot],
        outputs=[chatbot, text_input, audio_input]
    )
    
    clear_btn.click(lambda: ([], "", None), outputs=[chatbot, text_input, audio_input])


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=8000
    )
