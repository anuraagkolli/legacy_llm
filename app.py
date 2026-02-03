#!/usr/bin/env python3
"""
Streamlit app for COBOL to Python translation using fine-tuned model.
"""

import streamlit as st
from test import load_model, translate_cobol

# Page config
st.set_page_config(
    page_title="COBOL to Python Translator",
    page_icon="🔄",
    layout="wide"
)

# Cache the model loading
@st.cache_resource
def get_model():
    """Load and cache the model and tokenizer."""
    model, tokenizer = load_model("checkpoints/final", device="auto")
    return model, tokenizer


def main():
    st.title("COBOL to Python Translator")
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
