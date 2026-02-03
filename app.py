#!/usr/bin/env python3
"""
LegacyAI - Streamlit app for COBOL to Python translation using fine-tuned model.
"""

import os
import re
from pathlib import Path

import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model path: use environment variable or default to local checkpoint
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/final")

# Base model name
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# System prompt for translation
SYSTEM_PROMPT = (
    "You are an expert COBOL to Python translator. Your task is to convert "
    "COBOL programs into clean, idiomatic Python code that preserves the original "
    "program's functionality and logic. Use appropriate Python patterns like classes, "
    "dataclasses, context managers, and standard library modules."
)

# Page config
st.set_page_config(
    page_title="LegacyAI",
    page_icon="🔄",
    layout="wide"
)


def load_model(model_path: str, device: str = "auto"):
    """Load the fine-tuned model (base + LoRA adapters)."""
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            device_map = {"": "mps"}
            torch_dtype = torch.float32
        else:
            device_map = {"": "cpu"}
            torch_dtype = torch.float32
    else:
        device_map = {"": device}
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Check if this is a LoRA model (has adapter_config.json)
    adapter_config_path = Path(model_path) / "adapter_config.json"

    if adapter_config_path.exists():
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load as regular model (merged or full fine-tune)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def extract_python_code(response: str) -> str:
    """Extract Python code from the model's response."""
    # Try to extract code between ```python and ```
    pattern = r"```python\n?(.*?)```"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()

    # Try without language specifier
    pattern = r"```\n?(.*?)```"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()

    # Return as-is if no code blocks found
    return response.strip()


def translate_cobol(
    model,
    tokenizer,
    cobol_code: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> str:
    """Translate COBOL code to Python using the fine-tuned model."""

    # Format as chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Translate the following COBOL code to Python:\n\n```cobol\n{cobol_code}\n```"}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    # Extract Python code from response
    python_code = extract_python_code(response)

    return python_code


# Cache the model loading
@st.cache_resource
def get_model():
    """Load and cache the model and tokenizer."""
    model, tokenizer = load_model(MODEL_PATH, device="auto")
    return model, tokenizer


def main():
    st.title("LegacyAI")
    st.markdown("Translate COBOL code to Python using a fine-tuned LLM")

    # Load model
    try:
        with st.spinner("Loading model (this may take a moment on first run)..."):
            model, tokenizer = get_model()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.info("Make sure the model checkpoint exists at: checkpoints/final")
        return

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("COBOL Input")
        cobol_code = st.text_area(
            "Enter COBOL code:",
            height=400,
            placeholder="""       IDENTIFICATION DIVISION.
       PROGRAM-ID. EXAMPLE.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-NUM1         PIC 9(2) VALUE 10.
       01  WS-NUM2         PIC 9(2) VALUE 20.
       01  WS-RESULT       PIC 9(3) VALUE ZEROS.

       PROCEDURE DIVISION.
       0000-MAIN.
           ADD WS-NUM1 TO WS-NUM2 GIVING WS-RESULT
           DISPLAY "Result: " WS-RESULT
           STOP RUN.""",
            key="cobol_input"
        )

        # Translate button
        translate_button = st.button("🔄 Translate", type="primary", use_container_width=True)

    with col2:
        st.subheader("Python Output")

        # Initialize session state for storing translation
        if "python_output" not in st.session_state:
            st.session_state.python_output = ""

        # Perform translation
        if translate_button:
            if not cobol_code.strip():
                st.warning("Please enter some COBOL code to translate.")
            else:
                with st.spinner("Translating..."):
                    try:
                        python_code = translate_cobol(
                            model,
                            tokenizer,
                            cobol_code,
                            max_new_tokens=1024,
                            temperature=0.1,
                            top_p=0.95
                        )
                        st.session_state.python_output = python_code
                        st.success("Translation complete!")
                    except Exception as e:
                        st.error(f"Translation failed: {str(e)}")
                        st.info("Try checking your COBOL syntax or reducing the code length.")

        # Display output
        if st.session_state.python_output:
            st.code(st.session_state.python_output, language="python", line_numbers=True)
        else:
            st.info("Python code will appear here after translation.")

    # Footer with usage tips
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        - **Enter COBOL code** in the left pane
        - Click **Translate** to convert to Python
        - The Python output includes **syntax highlighting** and a **built-in copy button**
        - The model is cached after first load for faster subsequent translations
        - For best results, use complete COBOL code blocks with proper structure
        """)


if __name__ == "__main__":
    main()
