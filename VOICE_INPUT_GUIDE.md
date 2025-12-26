# Voice Input Integration Guide

## Basic Setup (Current Implementation)

I've updated your chatbot to support voice input by adding `multimodal=True` to the `gr.ChatInterface`. This enables an audio recording button in the chat interface.

### What Changed

In your notebook's Gradio cell (the one starting at line 271), replace the code with the content from `gradio_interface_with_audio.py`.

**Key changes:**

1. Added `multimodal=True` parameter to `gr.ChatInterface`
2. Updated the `chat()` function to handle both text and audio inputs
3. The message parameter can now be either a string or a dictionary with 'text' and 'files' keys

## Current Limitations

The current implementation shows the audio input button, but **audio transcription to text is not yet implemented**. To fully enable voice input, you'll need to add audio transcription.

## Adding Audio Transcription (Optional)

To convert voice recordings to text, you can use OpenAI's Whisper API:

### Option 1: Using OpenAI Whisper API

1. **Install the package:**

   ```bash
   pip install openai
   ```

2. **Update your chat function:**

```python
import gradio as gr
from openai import OpenAI
from langchain_core.messages import HumanMessage

client = OpenAI()  # Uses OPENAI_API_KEY from environment

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

def chat(message, history):
    try:
        # Handle multimodal input
        if isinstance(message, dict):
            user_message = message.get("text", "")
            audio_files = message.get("files", [])

            # If audio is provided, transcribe it
            if audio_files:
                audio_path = audio_files[0]  # Get first audio file
                transcribed_text = transcribe_audio(audio_path)

                if transcribed_text:
                    # Use transcribed text if no text was provided
                    user_message = user_message or transcribed_text
                else:
                    return "Sorry, I couldn't transcribe the audio. Please try again."
        else:
            user_message = message

        if not user_message:
            return "Please provide a message or voice input."

        # Invoke the graph with the message
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config
        )

        assistant_response = result["messages"][-1].content
        return assistant_response

    except Exception as e:
        print(f"Chat error: {e}")
        return f"Sorry, I encountered an error. -> {e}"

gr.ChatInterface(
    chat,
    title="Health Chatbot",
    description="Hi! What can I help you with? You can type or use voice input.",
    multimodal=True
).launch()
```

### Option 2: Using Local Whisper (Free, No API calls)

If you want to avoid API costs:

1. **Install whisper:**

   ```bash
   pip install openai-whisper
   ```

2. **Update transcription function:**

```python
import whisper

# Load model once (outside the function)
whisper_model = whisper.load_model("base")  # or "small", "medium", "large"

def transcribe_audio(audio_path):
    """Transcribe audio using local Whisper model."""
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Transcription error: {e}")
        return None
```

## Testing Voice Input

1. Run your updated notebook cell
2. Click the microphone/audio button in the chat interface
3. Record your question
4. The audio will be processed and (if transcription is set up) converted to text
5. The chatbot will respond as normal

## Notes

- The audio file format is handled automatically by Gradio
- Make sure your `.env` file has the `OPENAI_API_KEY` if using Whisper API
- Local Whisper models require more disk space but are free to use
- The "base" Whisper model is ~140MB, "small" is ~460MB

## Troubleshooting

- **No audio button appears:** Make sure `multimodal=True` is set in `gr.ChatInterface`
- **Audio not transcribing:** Check that the transcription function is properly configured
- **API errors:** Verify your OpenAI API key is set correctly in `.env`
