#!/usr/bin/env python3
"""Local chat UI backed by AirLLM (https://github.com/lyogavin/airllm), with file-upload-as-context."""
import os

import gradio as gr

MODEL_ID = os.environ.get("AIRLLM_MODEL_ID", "Qwen/Qwen3-8B")
COMPRESSION = os.environ.get("AIRLLM_COMPRESSION", "")  # "4bit"/"8bit" need bitsandbytes + CUDA; leave empty on CPU-only
MAX_INPUT_TOKENS = int(os.environ.get("AIRLLM_MAX_INPUT_TOKENS", "4096"))
MAX_NEW_TOKENS = int(os.environ.get("AIRLLM_MAX_NEW_TOKENS", "512"))
MAX_CONTEXT_CHARS = int(os.environ.get("AIRLLM_MAX_CONTEXT_CHARS", "12000"))
SYSTEM_PROMPT_FILE = "Agent.MD"

_model = None


def load_system_prompt():
    path = os.path.join(os.getcwd(), SYSTEM_PROMPT_FILE)
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except (FileNotFoundError, IOError):
        return "You are a helpful assistant."


def get_model():
    global _model
    if _model is None:
        from airllm import AutoModel
        print(f"Loading AirLLM model '{MODEL_ID}' (compression={COMPRESSION}) ...")
        _model = AutoModel.from_pretrained(MODEL_ID, compression=COMPRESSION or None)
        print("Model loaded.")
    return _model


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return f"[Could not read {os.path.basename(file_path)}: install 'pypdf' to extract PDF text]"
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    try:
        with open(file_path, "r", errors="ignore") as f:
            return f.read()
    except (IOError, UnicodeDecodeError) as e:
        return f"[Could not read {os.path.basename(file_path)}: {e}]"


def build_context_block(files):
    if not files:
        return ""
    blocks = []
    for fp in files:
        name = os.path.basename(fp)
        text = extract_text(fp)
        blocks.append(f"--- Begin file: {name} ---\n{text}\n--- End file: {name} ---")
    context = "\n\n".join(blocks)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n[... context truncated ...]"
    return context


def build_prompt(system_prompt, context, history, user_message):
    parts = [f"[system]\n{system_prompt}\n"]
    if context:
        parts.append(f"[context: uploaded files]\n{context}\n")
    for msg in history:
        tag = "user" if msg["role"] == "user" else "assistant"
        parts.append(f"[{tag}]\n{msg['content']}\n")
    parts.append(f"[user]\n{user_message}\n")
    parts.append("[assistant]\n")
    return "\n".join(parts)


def generate_reply(prompt):
    model = get_model()
    tokenizer = model.tokenizer
    input_tokens = tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=False,
    )
    input_ids = input_tokens["input_ids"]
    try:
        input_ids = input_ids.cuda()
    except (AssertionError, RuntimeError):
        pass  # no GPU available; AirLLM uses whatever device it was configured for

    output = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        return_dict_in_generate=True,
    )
    generated = output.sequences[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def chat_fn(user_message, chat_history, uploaded_files, context_state):
    if not user_message or not user_message.strip():
        return chat_history, chat_history, "", context_state

    if uploaded_files:
        context_state = build_context_block(uploaded_files)

    try:
        system_prompt = load_system_prompt()
        prompt = build_prompt(system_prompt, context_state or "", chat_history, user_message)
        reply = generate_reply(prompt)
    except Exception as e:
        reply = f"Error: {e}"

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return chat_history, chat_history, "", context_state


def clear_fn():
    return [], [], None, ""


with gr.Blocks(title="AirLLM Local Chat") as demo:
    gr.Markdown(f"# AirLLM Local Chat\nModel: `{MODEL_ID}` | Compression: `{COMPRESSION}`")

    chatbot = gr.Chatbot(height=500)
    history_state = gr.State([])
    context_state = gr.State("")

    file_upload = gr.File(label="Upload files as context (txt/md/code/pdf)", file_count="multiple", type="filepath")

    with gr.Row():
        msg = gr.Textbox(label="Message", placeholder="Type your message...", scale=4)
        send = gr.Button("Send", scale=1)

    clear = gr.Button("Clear chat")

    send.click(
        chat_fn,
        [msg, history_state, file_upload, context_state],
        [chatbot, history_state, msg, context_state],
    )
    msg.submit(
        chat_fn,
        [msg, history_state, file_upload, context_state],
        [chatbot, history_state, msg, context_state],
    )
    clear.click(clear_fn, None, [chatbot, history_state, file_upload, context_state])

if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
    )
