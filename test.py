#!/usr/bin/env python3
"""
Inference Script for COBOL->Python Translation
Tests the fine-tuned model on new COBOL code.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

console = Console()

# Base model name
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# System prompt for translation
SYSTEM_PROMPT = (
    "You are an expert COBOL to Python translator. Your task is to convert "
    "COBOL programs into clean, idiomatic Python code that preserves the original "
    "program's functionality and logic. Use appropriate Python patterns like classes, "
    "dataclasses, context managers, and standard library modules."
)


def load_model(model_path: str, device: str = "auto"):
    """Load the fine-tuned model (base + LoRA adapters)."""
    console.print(f"[blue]Loading model from {model_path}...[/blue]")

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
        console.print("[yellow]Loading LoRA adapter...[/yellow]")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)
        console.print("[green]LoRA adapter loaded successfully[/green]")
    else:
        # Load as regular model (merged or full fine-tune)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        console.print("[green]Model loaded successfully[/green]")

    model.eval()

    return model, tokenizer


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


def interactive_mode(model, tokenizer):
    """Run interactive translation mode."""
    console.print("\n[bold green]Interactive COBOL->Python Translation[/bold green]")
    console.print("Enter COBOL code (end with an empty line or 'END'):")
    console.print("Type 'quit' or 'exit' to stop.\n")

    while True:
        # Collect multi-line input
        lines = []
        console.print("[cyan]COBOL>[/cyan] ", end="")

        while True:
            try:
                line = input()
            except EOFError:
                break

            if line.lower() in ['quit', 'exit']:
                console.print("[yellow]Goodbye![/yellow]")
                return

            if line.strip() == "" and lines:
                break
            if line.strip().upper() == "END" and lines:
                break

            lines.append(line)

        if not lines:
            continue

        cobol_code = "\n".join(lines)

        console.print("\n[blue]Translating...[/blue]")

        try:
            python_code = translate_cobol(model, tokenizer, cobol_code)

            console.print("\n[bold green]Python Translation:[/bold green]")
            syntax = Syntax(python_code, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Generated Python", border_style="green"))
            console.print()

        except Exception as e:
            console.print(f"[red]Error during translation: {e}[/red]")

        console.print("-" * 60)


def translate_file(model, tokenizer, input_file: str, output_file: Optional[str] = None):
    """Translate a COBOL file to Python."""
    console.print(f"[blue]Reading COBOL code from {input_file}...[/blue]")

    with open(input_file, 'r') as f:
        cobol_code = f.read()

    console.print("[blue]Translating...[/blue]")
    python_code = translate_cobol(model, tokenizer, cobol_code)

    # Display result
    console.print("\n[bold green]Python Translation:[/bold green]")
    syntax = Syntax(python_code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Generated Python", border_style="green"))

    # Save if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(python_code)
        console.print(f"\n[green]Translation saved to {output_file}[/green]")

    return python_code


def run_example(model, tokenizer):
    """Run translation on a sample COBOL program."""
    example_cobol = '''       IDENTIFICATION DIVISION.
       PROGRAM-ID. HELLO.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-NAME              PIC X(20) VALUE SPACES.
       01  WS-GREETING          PIC X(50) VALUE SPACES.

       PROCEDURE DIVISION.
       0000-MAIN.
           DISPLAY "Enter your name: "
           ACCEPT WS-NAME
           STRING "Hello, " DELIMITED SIZE
                  WS-NAME DELIMITED SPACES
                  "!" DELIMITED SIZE
                  INTO WS-GREETING
           END-STRING
           DISPLAY WS-GREETING
           STOP RUN.'''

    console.print("[bold]Example COBOL Input:[/bold]")
    syntax = Syntax(example_cobol, "cobol", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="COBOL Code", border_style="blue"))

    console.print("\n[blue]Translating...[/blue]\n")
    python_code = translate_cobol(model, tokenizer, example_cobol)

    console.print("[bold green]Python Translation:[/bold green]")
    syntax = Syntax(python_code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Generated Python", border_style="green"))


def main():
    parser = argparse.ArgumentParser(description="Test COBOL->Python translation model")
    parser.add_argument("--model", type=str, default="checkpoints/final",
                        help="Path to fine-tuned model")
    parser.add_argument("--input", type=str, default=None,
                        help="COBOL file to translate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for Python translation")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--example", action="store_true",
                        help="Run example translation")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to use for inference")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens to generate")

    args = parser.parse_args()

    console.print("[bold green]" + "=" * 60)
    console.print("[bold green]COBOL->Python Translation Tester[/bold green]")
    console.print("[bold green]" + "=" * 60)

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    if args.example:
        run_example(model, tokenizer)
    elif args.interactive:
        interactive_mode(model, tokenizer)
    elif args.input:
        translate_file(model, tokenizer, args.input, args.output)
    else:
        # Default: run example then interactive
        run_example(model, tokenizer)
        console.print("\n" + "=" * 60 + "\n")
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
