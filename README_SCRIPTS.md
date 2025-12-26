# Health Chatbot - Python Scripts

Your Jupyter notebook has been converted into two modular Python scripts:

## 📁 File Structure

```
Health Chatbot/
├── agent.py          # LangGraph agent logic & workflow
├── gradio_app.py     # Gradio web interface with voice input
├── .env              # Environment variables (API keys)
└── requirements.txt  # Python dependencies
```

## 🔗 How They Link Together

### `agent.py` (Backend)

Contains all the LangGraph workflow logic:

- Agent state management
- Tool definitions (general_search)
- System prompts
- Graph compilation
- **Main function**: `get_chatbot_response(user_message, thread_id)`

### `gradio_app.py` (Frontend)

Provides the web interface:

- Imports `get_chatbot_response` from `agent.py`
- Handles user input (text + voice)
- Audio transcription (optional)
- Gradio ChatInterface with multimodal support

**Connection**: `gradio_app.py` imports and calls `get_chatbot_response()` from `agent.py`

## 🚀 Running the Application

### Option 1: Run Gradio Interface (Recommended)

```bash
python gradio_app.py
```

Then open http://127.0.0.1:7860 in your browser.

### Option 2: Test Agent Directly

```bash
python agent.py
```

This runs a simple CLI test of the agent.

### Option 3: Import in Your Own Code

```python
from agent import get_chatbot_response

response = get_chatbot_response("What are flu symptoms?", thread_id="user_123")
print(response)
```

## 🎤 Enabling Voice Input

The voice input button is already visible in the interface, but transcription needs setup:

1. **Install OpenAI package:**

   ```bash
   pip install openai
   ```

2. **Add your API key to `.env`:**

   ```
   OPENAI_API_KEY=your-key-here
   ```

3. **Uncomment the transcription code in `gradio_app.py`:**

   - Line 8: Uncomment `from openai import OpenAI`
   - Line 9: Uncomment `client = OpenAI()`
   - Lines 19-27: Uncomment the transcription implementation

4. **Restart the app:**
   ```bash
   python gradio_app.py
   ```

Now you can click the microphone button and speak your questions!

## 📦 Dependencies

Make sure these are in your `requirements.txt`:

```
gradio
python-dotenv
langgraph
langchain
langchain-openai
langchain-tavily
openai  # Optional, for voice input
```

Install with:

```bash
pip install -r requirements.txt
```

## ✅ Benefits of This Structure

1. **Separation of Concerns**: Agent logic separate from UI
2. **Reusability**: Use `agent.py` in other projects
3. **Testability**: Test agent without running Gradio
4. **Maintainability**: Easier to update each component
5. **Flexibility**: Swap Gradio for Streamlit, FastAPI, etc.

## 🧪 Testing

Test the agent:

```bash
python agent.py
```

Test the full interface:

```bash
python gradio_app.py
```

## 🔧 Customization

### Change the Model

In `agent.py`, line 62:

```python
llm = ChatOpenAI(model="gpt-4o-mini")  # Change to gpt-4, etc.
```

### Modify the System Prompt

In `agent.py`, the `SYSTEM_MESSAGE` variable (lines 45-59)

### Adjust Interface Theme

In `gradio_app.py`, line 82:

```python
theme=gr.themes.Soft(),  # Try Base(), Glass(), Monochrome()
```

## 📝 Notes

- Both scripts use the same `.env` file for API keys
- Conversation history persists within a session (thread_id)
- The agent uses InMemorySaver for checkpointing (resets on restart)
- Voice input recording happens in the browser (no mic permissions needed on server)
