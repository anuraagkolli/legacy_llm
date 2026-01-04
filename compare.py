#!/usr/bin/env python3
"""
Quick Comparison Tool for COBOL->Python Translation

Compare fine-tuned vs base model on a single COBOL sample interactively.
"""

import argparse
import ast
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.columns import Columns

console = Console()

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert COBOL to Python translator. Your task is to convert "
    "COBOL programs into clean, idiomatic Python code that preserves the original "
    "program's functionality and logic. Use appropriate Python patterns like classes, "
    "dataclasses, context managers, and standard library modules."
)

EXAMPLE_COBOL = '''       IDENTIFICATION DIVISION.
       PROGRAM-ID. CALCULATE-DISCOUNT.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-PRICE            PIC 9(5)V99 VALUE ZEROS.
       01  WS-QUANTITY         PIC 9(3) VALUE ZEROS.
       01  WS-TOTAL            PIC 9(7)V99 VALUE ZEROS.
       01  WS-DISCOUNT-PCT     PIC 9V99 VALUE ZEROS.
       01  WS-DISCOUNT-AMT     PIC 9(6)V99 VALUE ZEROS.
       01  WS-FINAL-TOTAL      PIC 9(7)V99 VALUE ZEROS.

       PROCEDURE DIVISION.
       0000-MAIN.
           PERFORM 1000-GET-INPUT
           PERFORM 2000-CALCULATE
           PERFORM 3000-DISPLAY-RESULTS
           STOP RUN.

       1000-GET-INPUT.
           DISPLAY "Enter price: "
           ACCEPT WS-PRICE
           DISPLAY "Enter quantity: "
           ACCEPT WS-QUANTITY.

       2000-CALCULATE.
           COMPUTE WS-TOTAL = WS-PRICE * WS-QUANTITY
           EVALUATE TRUE
               WHEN WS-TOTAL >= 1000
                   MOVE 0.15 TO WS-DISCOUNT-PCT
               WHEN WS-TOTAL >= 500
                   MOVE 0.10 TO WS-DISCOUNT-PCT
               WHEN WS-TOTAL >= 100
                   MOVE 0.05 TO WS-DISCOUNT-PCT
               WHEN OTHER
                   MOVE 0 TO WS-DISCOUNT-PCT
           END-EVALUATE
           COMPUTE WS-DISCOUNT-AMT = WS-TOTAL * WS-DISCOUNT-PCT
           COMPUTE WS-FINAL-TOTAL = WS-TOTAL - WS-DISCOUNT-AMT.

       3000-DISPLAY-RESULTS.
           DISPLAY "Subtotal: $" WS-TOTAL
           DISPLAY "Discount: " WS-DISCOUNT-PCT "%"
           DISPLAY "Discount Amount: $" WS-DISCOUNT-AMT
           DISPLAY "Final Total: $" WS-FINAL-TOTAL.'''


def get_device_config():
    """Determine device configuration."""
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device_map = {"": "mps"}
        torch_dtype = torch.float32
    else:
        device_map = {"": "cpu"}
        torch_dtype = torch.float32
    return device_map, torch_dtype


def extract_python_code(response: str) -> str:
    """Extract Python code from model response."""
    pattern = r"```python\n?(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern = r"```\n?(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response.strip()


def translate(model, tokenizer, cobol_code: str, max_new_tokens: int = 1024, temperature: float = 0.1) -> str:
    """Translate COBOL code to Python."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Translate the following COBOL code to Python:\n\n```cobol\n{cobol_code}\n```"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return extract_python_code(response)


def check_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Check if Python code is syntactically valid."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def load_models(model_path: str, load_base: bool = True):
    """Load both models."""
    device_map, torch_dtype = get_device_config()

    models = {}

    # Load base model
    if load_base:
        console.print("[blue]Loading base model...[/blue]")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        base_model.eval()
        models['base'] = (base_model, tokenizer)
        console.print("[green]Base model loaded[/green]")

    # Load fine-tuned model
    console.print(f"[blue]Loading fine-tuned model from {model_path}...[/blue]")
    ft_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    ft_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_base, model_path)
    ft_model.eval()
    models['finetuned'] = (ft_model, ft_tokenizer)
    console.print("[green]Fine-tuned model loaded[/green]")

    return models


def compare_single(models: dict, cobol_code: str, show_input: bool = True):
    """Compare outputs from both models on a single input."""

    if show_input:
        console.print("\n[bold cyan]COBOL Input:[/bold cyan]")
        console.print(Panel(Syntax(cobol_code, "cobol", theme="monokai", line_numbers=True),
                          border_style="blue", title="Input"))

    results = {}

    for name, (model, tokenizer) in models.items():
        label = "Base Model" if name == "base" else "Fine-tuned"
        console.print(f"\n[yellow]Generating with {label}...[/yellow]")

        try:
            output = translate(model, tokenizer, cobol_code)
            valid, error = check_syntax(output)
        except Exception as e:
            output = f"ERROR: {e}"
            valid, error = False, str(e)

        results[name] = {
            'output': output,
            'valid': valid,
            'error': error,
        }

    # Display results side by side
    console.print("\n" + "="*80)
    console.print("[bold]COMPARISON RESULTS[/bold]")
    console.print("="*80 + "\n")

    panels = []

    if 'base' in results:
        base_status = "[green]✓ Valid[/green]" if results['base']['valid'] else f"[red]✗ Invalid[/red]"
        panels.append(Panel(
            Syntax(results['base']['output'], "python", theme="monokai", line_numbers=True, word_wrap=True),
            title=f"[yellow]Base Model[/yellow] ({base_status})",
            border_style="yellow",
            width=80,
        ))

    ft_status = "[green]✓ Valid[/green]" if results['finetuned']['valid'] else f"[red]✗ Invalid[/red]"
    panels.append(Panel(
        Syntax(results['finetuned']['output'], "python", theme="monokai", line_numbers=True, word_wrap=True),
        title=f"[green]Fine-tuned[/green] ({ft_status})",
        border_style="green",
        width=80,
    ))

    for panel in panels:
        console.print(panel)
        console.print()

    # Summary table
    if 'base' in results:
        table = Table(title="Quick Comparison", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Base Model", justify="center")
        table.add_column("Fine-tuned", justify="center")

        table.add_row(
            "Valid Python",
            "[green]Yes[/green]" if results['base']['valid'] else "[red]No[/red]",
            "[green]Yes[/green]" if results['finetuned']['valid'] else "[red]No[/red]",
        )
        table.add_row(
            "Output Length",
            str(len(results['base']['output'])),
            str(len(results['finetuned']['output'])),
        )
        table.add_row(
            "Line Count",
            str(len(results['base']['output'].split('\n'))),
            str(len(results['finetuned']['output'].split('\n'))),
        )

        console.print(table)

    return results


def interactive_mode(models: dict):
    """Run interactive comparison mode."""
    console.print("\n[bold green]Interactive Comparison Mode[/bold green]")
    console.print("Enter COBOL code (end with empty line or 'END'), or type 'example' for sample code")
    console.print("Type 'quit' to exit.\n")

    while True:
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

            if line.lower() == 'example':
                compare_single(models, EXAMPLE_COBOL)
                break

            if line.strip() == "" and lines:
                break
            if line.strip().upper() == "END" and lines:
                break

            lines.append(line)

        if lines:
            cobol_code = "\n".join(lines)
            compare_single(models, cobol_code)

        console.print("\n" + "-"*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Quick comparison of fine-tuned vs base model")
    parser.add_argument("--model", type=str, default="checkpoints/final",
                        help="Path to fine-tuned model")
    parser.add_argument("--input", type=str, default=None,
                        help="COBOL file to translate")
    parser.add_argument("--example", action="store_true",
                        help="Run on built-in example")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--finetuned-only", action="store_true",
                        help="Only load and test fine-tuned model")

    args = parser.parse_args()

    console.print("[bold green]" + "="*60)
    console.print("[bold green]COBOL->Python Model Comparison[/bold green]")
    console.print("[bold green]" + "="*60 + "\n")

    # Load models
    models = load_models(args.model, load_base=not args.finetuned_only)

    if args.example:
        compare_single(models, EXAMPLE_COBOL)
    elif args.input:
        with open(args.input, 'r') as f:
            cobol_code = f.read()
        compare_single(models, cobol_code)
    elif args.interactive:
        interactive_mode(models)
    else:
        # Default: run example then interactive
        compare_single(models, EXAMPLE_COBOL)
        console.print("\n" + "="*60 + "\n")
        interactive_mode(models)


if __name__ == "__main__":
    main()
