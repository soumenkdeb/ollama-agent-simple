#!/usr/bin/env python3
import requests
import sys
import json
import os

# Get the current working directory (where the script is running)
current_dir = os.getcwd()
# Specify your filename (replace 'your_file.txt' with your actual filename)
filename = 'Agent.MD'
# Construct the full file path
file_path = os.path.join(current_dir, filename)


try:
    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read()
        print("File content:")
        print(content)
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found in the current directory.")
except IOError:
    print(f"Error: Could not read the file '{filename}'.")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")


OLLAMA_URL = "http://localhost:11434"
MODEL = "ministral-3"  # or gemma3:latest, ministral-3,"llama3.2:latest" "deepseek-v2:latest", "gpt-oss:latest", etc.

def build_prompt(conversation):
    """Turn chat-style messages into a single prompt."""
    parts = []
    for m in conversation:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[system]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[assistant]\n{content}\n")
        else:
            parts.append(f"[user]\n{content}\n")
    parts.append("[assistant]\n")
    return "\n".join(parts)

def generate_stream(conversation, model=MODEL):
    url = f"{OLLAMA_URL}/api/generate"
    prompt = build_prompt(conversation)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    with requests.post(url, json=payload, stream=True, timeout=1800) as resp:
        resp.raise_for_status()
        full_text = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # /api/generate: {"response": "...", "done": bool, ...}
            if "response" in data:
                chunk = data["response"]
                full_text += chunk
                print(chunk, end="", flush=True)

            if data.get("done"):
                break

        print()
        return full_text

def main():
    print(f"Starting streaming agent with Ollama model: {MODEL}")
    print("Type 'exit' or Ctrl+C to quit.\n")

    conversation = [
        {
            "role": "system",
            "content": content
        }
    ]

    while True:
        try:
            user_input = input(MODEL+" > ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Bye.")
                break

            conversation.append({"role": "user", "content": user_input})
            reply = generate_stream(conversation)
            conversation.append({"role": "assistant", "content": reply})
            print()

        except KeyboardInterrupt:
            print("\nInterrupted, exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()