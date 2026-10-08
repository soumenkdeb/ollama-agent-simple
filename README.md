# Local LLM Agents

Two standalone local-LLM apps in this repo:

1.  **`ollama_agent_stream.py`** — CLI chat via [Ollama](https://ollama.com/), streaming output.
2.  **`airllm_chat_app.py`** — Local web chat UI (Gradio) via [AirLLM](https://github.com/lyogavin/airllm), with file-upload-as-context.

Both load the system prompt from `Agent.MD` in the repo root.

---

## 1. Ollama CLI Chat (`ollama_agent_stream.py`)

A lightweight command-line interface for interacting with local LLMs via Ollama. Supports streaming responses and multi-turn conversation history.

### Features

-   **Streaming Responses**: Receive output token-by-token as it's generated.
-   **Context Awareness**: Maintains conversation history for multi-turn chats.
-   **Custom System Prompt**: Loads the system persona/instructions from `Agent.MD`.

### Prerequisites

1.  Python 3.
2.  **Ollama** running locally.
    -   Download from [ollama.com](https://ollama.com).
    -   Ensure the service is running (default `http://localhost:11434`).
3.  Python library: `requests`.

### Install

```bash
pip install requests
```

### Setup

1.  **System prompt**: create/edit `Agent.MD` in the repo root with the AI's persona/instructions, e.g.:

    ```text
    # Agent.MD example
    You are a helpful assistant who answers questions concisely.
    ```

2.  **Model**: open `ollama_agent_stream.py` and set `MODEL` to a model you've pulled in Ollama:

    ```python
    MODEL = "ministral-3"  # or "llama3", "mistral", "gemma3:latest", etc.
    ```

    List your pulled models with:

    ```bash
    ollama list
    ```

### Usage

```bash
python3 ollama_agent_stream.py
```

-   Type your message and press **Enter** to chat.
-   Type `exit` or `quit` (or press **Ctrl+C**) to stop.

### Configuration

Constants at the top of `ollama_agent_stream.py`:

-   `OLLAMA_URL`: URL of your Ollama instance (default: `http://localhost:11434`).
-   `MODEL`: the LLM to use.
-   `filename`: name of the system-prompt file (default: `Agent.MD`).

---

## 2. AirLLM Web Chat UI (`airllm_chat_app.py`)

A local Gradio web chat UI backed by AirLLM instead of Ollama. AirLLM streams model layers from disk so large Hugging Face models can run on limited VRAM. Supports uploading files whose text content gets injected as context for the chat.

### Prerequisites

1.  Python 3 (see note below on Python 3.14).
2.  A Hugging Face model id to load (default: `Qwen/Qwen3-8B`).
3.  **GPU note**: AirLLM officially supports NVIDIA CUDA and Apple Silicon only — no AMD/ROCm backend. On AMD GPUs (including iGPUs like the Radeon 680M) it runs **CPU-only**, which is slow for an 8B model. If you have an AMD GPU, consider using `ollama_agent_stream.py` instead (Ollama's llama.cpp backend has real AMD support).

### Install

```bash
pip install -r requirements-airllm.txt
```

> **Note:** This repo's venv is on Python 3.14, which is very new — `torch`/`airllm`/`bitsandbytes` wheels may lag behind. If install fails, create a separate venv on Python 3.11/3.12 for this app.

### Setup

Same `Agent.MD` system prompt file as the Ollama script (optional — falls back to a generic assistant prompt if missing).

### Configuration (environment variables)

-   `AIRLLM_MODEL_ID`: HF repo id to load (default: `Qwen/Qwen3-8B`).
-   `AIRLLM_COMPRESSION`: Quantization level — unset/empty (default, full precision) or `4bit`/`8bit` (needs `bitsandbytes` + a CUDA GPU; not useful on CPU-only setups like AMD iGPUs).
-   `AIRLLM_MAX_INPUT_TOKENS`: Max prompt tokens (default: `4096`).
-   `AIRLLM_MAX_NEW_TOKENS`: Max tokens generated per reply (default: `512`).
-   `AIRLLM_MAX_CONTEXT_CHARS`: Max chars of uploaded-file context kept (default: `12000`).
-   `GRADIO_SERVER_NAME` / `GRADIO_PORT`: Bind address/port (default: `127.0.0.1:7860`).

### Usage

```bash
python3 airllm_chat_app.py
```

Open the printed local URL in a browser:

-   Type a message and press **Enter** (or click **Send**) to chat. First message triggers a one-time model load (can take a while on CPU).
-   Upload `.txt`/`.md`/code files or PDFs (PDF needs `pypdf`) via the file box — their extracted text is injected as context for subsequent messages.
-   Click **Clear chat** to reset the conversation and uploaded context.
