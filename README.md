# Ollama Agent Stream

A simple, lightweight Python command-line interface for interacting with local LLMs via [Ollama](https://ollama.com/). This agent supports streaming responses and maintains conversation context.

## Features

- **Streaming Responses**: Receive output token-by-token as it's generated.
- **Context Awareness**: Maintains conversation history for multi-turn chats.
- **Custom System Prompt**: Loads the system persona/instructions from an external `Agent.MD` file.
- **Easy Configuration**: Simple variable tweaks to change models or API endpoints.

## Prerequisites

1.  **Python 3**: Ensure you have Python installed.
2.  **Ollama**: strictly requires a local instance of Ollama running.
    -   Download from [ollama.com](https://ollama.com).
    -   Ensure the service is running (default `http://localhost:11434`).
3.  **Python Libraries**:
    -   `requests`

## Installation

1.  Clone this repository or download the script.
2.  Install the required dependency:
    ```bash
    pip install requests
    ```

## Setup

1.  **Prepare the System Prompt**:
    Create a file named `Agent.MD` in the same directory as the script. This file should contain the system instructions or "persona" for the AI (e.g., "You are a helpful coding assistant...").
    
    ```text
    # Agent.MD example
    You are a helpful assistant who answers questions concisely.
    ```

2.  **Check the Model**:
    Open `ollama_agent_stream.py` and verify the `MODEL` variable matches a model you have pulled in Ollama.
    ```python
    MODEL = "ministral-3" # Change this to "llama3", "mistral", "gemma", etc.
    ```
    To see your available models, run:
    ```bash
    ollama list
    ```

## Usage

Run the script from your terminal:

```bash
python3 ollama_agent_stream.py
```

Once running:
-   Type your message and press **Enter** to chat.
-   Type `exit` or `quit` (or press `Ctrl+C`) to stop the session.

## Configuration

You can modify the following constants at the top of `ollama_agent_stream.py` to customize behavior:

-   `OLLAMA_URL`: The URL of your Ollama instance (default: `http://localhost:11434`).
-   `MODEL`: The specific LLM to use.
-   `filename`: The name of the file containing the system prompt (default: `Agent.MD`).
