#!/usr/bin/env python3
"""
Evaluation Script for COBOL->Python Translation Model

Compares fine-tuned model against base model on validation data.
Computes BLEU scores, syntax validity, and generates side-by-side comparisons.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

console = Console()

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert COBOL to Python translator. Your task is to convert "
    "COBOL programs into clean, idiomatic Python code that preserves the original "
    "program's functionality and logic. Use appropriate Python patterns like classes, "
    "dataclasses, context managers, and standard library modules."
)


def load_base_model(device_map, torch_dtype):
    """Load the base model without LoRA adapters."""
    console.print("[blue]Loading base model (unfinetuned)...[/blue]")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    console.print("[green]Base model loaded[/green]")
    return model, tokenizer


def load_finetuned_model(model_path: str, device_map, torch_dtype):
    """Load the fine-tuned model with LoRA adapters."""
    console.print(f"[blue]Loading fine-tuned model from {model_path}...[/blue]")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    console.print("[green]Fine-tuned model loaded[/green]")
    return model, tokenizer


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


def compute_bleu(reference: str, candidate: str) -> float:
    """Compute BLEU score between reference and candidate."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    if not cand_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    try:
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    except:
        return 0.0


def compute_code_bleu(reference: str, candidate: str) -> float:
    """Compute a code-aware BLEU score (tokenizes on code boundaries)."""
    def tokenize_code(code: str) -> list[str]:
        # Split on whitespace, punctuation, and operators
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return [t.lower() for t in tokens if t.strip()]

    ref_tokens = tokenize_code(reference)
    cand_tokens = tokenize_code(candidate)

    if not cand_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    try:
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    except:
        return 0.0


def load_validation_data(val_path: str, limit: Optional[int] = None) -> list[dict]:
    """Load validation data from JSONL file."""
    samples = []
    with open(val_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line)
            messages = data['messages']

            # Extract COBOL from user message
            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
            cobol_match = re.search(r'```cobol\n(.*?)```', user_msg, re.DOTALL)
            cobol = cobol_match.group(1) if cobol_match else ''

            # Extract reference Python from assistant message
            assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
            reference = extract_python_code(assistant_msg)

            samples.append({
                'cobol': cobol,
                'reference': reference,
            })

    return samples


def evaluate_model(model, tokenizer, samples: list[dict], model_name: str) -> list[dict]:
    """Evaluate a model on all samples."""
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Evaluating {model_name}...", total=len(samples))

        for sample in samples:
            try:
                output = translate(model, tokenizer, sample['cobol'])
            except Exception as e:
                output = f"ERROR: {e}"

            valid, error = check_syntax(output) if not output.startswith("ERROR:") else (False, output)
            bleu = compute_bleu(sample['reference'], output)
            code_bleu = compute_code_bleu(sample['reference'], output)

            results.append({
                'cobol': sample['cobol'],
                'reference': sample['reference'],
                'output': output,
                'valid_syntax': valid,
                'syntax_error': error,
                'bleu': bleu,
                'code_bleu': code_bleu,
            })

            progress.advance(task)

    return results


def print_summary(base_results: list[dict], finetuned_results: list[dict]):
    """Print summary statistics."""
    def compute_stats(results: list[dict]) -> dict:
        valid_count = sum(1 for r in results if r['valid_syntax'])
        return {
            'count': len(results),
            'valid_syntax': valid_count,
            'valid_pct': 100 * valid_count / len(results) if results else 0,
            'avg_bleu': sum(r['bleu'] for r in results) / len(results) if results else 0,
            'avg_code_bleu': sum(r['code_bleu'] for r in results) / len(results) if results else 0,
        }

    base_stats = compute_stats(base_results)
    ft_stats = compute_stats(finetuned_results)

    table = Table(title="Evaluation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Base Model", justify="right")
    table.add_column("Fine-tuned", justify="right")
    table.add_column("Improvement", justify="right")

    def fmt_improvement(base_val, ft_val, higher_is_better=True):
        diff = ft_val - base_val
        if higher_is_better:
            color = "green" if diff > 0 else "red" if diff < 0 else "white"
        else:
            color = "red" if diff > 0 else "green" if diff < 0 else "white"
        sign = "+" if diff > 0 else ""
        return f"[{color}]{sign}{diff:.2f}[/{color}]"

    table.add_row(
        "Samples Evaluated",
        str(base_stats['count']),
        str(ft_stats['count']),
        "-"
    )
    table.add_row(
        "Valid Python Syntax",
        f"{base_stats['valid_pct']:.1f}%",
        f"{ft_stats['valid_pct']:.1f}%",
        fmt_improvement(base_stats['valid_pct'], ft_stats['valid_pct'])
    )
    table.add_row(
        "Average BLEU",
        f"{base_stats['avg_bleu']:.4f}",
        f"{ft_stats['avg_bleu']:.4f}",
        fmt_improvement(base_stats['avg_bleu'], ft_stats['avg_bleu'])
    )
    table.add_row(
        "Average CodeBLEU",
        f"{base_stats['avg_code_bleu']:.4f}",
        f"{ft_stats['avg_code_bleu']:.4f}",
        fmt_improvement(base_stats['avg_code_bleu'], ft_stats['avg_code_bleu'])
    )

    console.print()
    console.print(table)
    console.print()


def print_side_by_side(base_results: list[dict], finetuned_results: list[dict], num_examples: int = 3):
    """Print side-by-side comparison of outputs."""
    console.print("\n[bold]Side-by-Side Comparisons[/bold]\n")

    for i in range(min(num_examples, len(base_results))):
        base = base_results[i]
        ft = finetuned_results[i]

        console.print(f"[bold cyan]{'='*80}[/bold cyan]")
        console.print(f"[bold]Example {i+1}[/bold]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")

        # Show COBOL input (truncated)
        cobol_preview = base['cobol'][:500] + "..." if len(base['cobol']) > 500 else base['cobol']
        console.print("[bold]COBOL Input:[/bold]")
        console.print(Panel(Syntax(cobol_preview, "cobol", theme="monokai"), border_style="blue"))

        # Show base model output
        base_status = "[green]Valid[/green]" if base['valid_syntax'] else f"[red]Invalid: {base['syntax_error'][:50]}[/red]"
        console.print(f"\n[bold]Base Model Output[/bold] (BLEU: {base['bleu']:.4f}, Syntax: {base_status}):")
        console.print(Panel(Syntax(base['output'][:1000], "python", theme="monokai"), border_style="yellow"))

        # Show fine-tuned output
        ft_status = "[green]Valid[/green]" if ft['valid_syntax'] else f"[red]Invalid: {ft['syntax_error'][:50]}[/red]"
        console.print(f"\n[bold]Fine-tuned Output[/bold] (BLEU: {ft['bleu']:.4f}, Syntax: {ft_status}):")
        console.print(Panel(Syntax(ft['output'][:1000], "python", theme="monokai"), border_style="green"))

        # Show reference
        console.print("\n[bold]Reference (from training data):[/bold]")
        console.print(Panel(Syntax(base['reference'][:1000], "python", theme="monokai"), border_style="cyan"))
        console.print()


def save_results(base_results: list[dict], finetuned_results: list[dict], output_path: str):
    """Save detailed results to JSON file."""
    results = {
        'base_model': BASE_MODEL_NAME,
        'base_results': base_results,
        'finetuned_results': finetuned_results,
        'summary': {
            'base': {
                'valid_syntax_pct': 100 * sum(1 for r in base_results if r['valid_syntax']) / len(base_results),
                'avg_bleu': sum(r['bleu'] for r in base_results) / len(base_results),
                'avg_code_bleu': sum(r['code_bleu'] for r in base_results) / len(base_results),
            },
            'finetuned': {
                'valid_syntax_pct': 100 * sum(1 for r in finetuned_results if r['valid_syntax']) / len(finetuned_results),
                'avg_bleu': sum(r['bleu'] for r in finetuned_results) / len(finetuned_results),
                'avg_code_bleu': sum(r['code_bleu'] for r in finetuned_results) / len(finetuned_results),
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]Detailed results saved to {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate COBOL->Python translation model")
    parser.add_argument("--model", type=str, default="checkpoints/final",
                        help="Path to fine-tuned model")
    parser.add_argument("--val-data", type=str, default="data/processed/val.jsonl",
                        help="Path to validation data")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max samples to evaluate (default: 20, use -1 for all)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for detailed results")
    parser.add_argument("--examples", type=int, default=3,
                        help="Number of side-by-side examples to show")
    parser.add_argument("--finetuned-only", action="store_true",
                        help="Only evaluate fine-tuned model (skip base model)")

    args = parser.parse_args()

    console.print("[bold green]" + "="*60)
    console.print("[bold green]COBOL->Python Model Evaluation[/bold green]")
    console.print("[bold green]" + "="*60 + "\n")

    # Load validation data
    limit = None if args.limit == -1 else args.limit
    console.print(f"[blue]Loading validation data from {args.val_data}...[/blue]")
    samples = load_validation_data(args.val_data, limit)
    console.print(f"[green]Loaded {len(samples)} samples[/green]\n")

    device_map, torch_dtype = get_device_config()

    # Evaluate base model
    if not args.finetuned_only:
        base_model, base_tokenizer = load_base_model(device_map, torch_dtype)
        base_results = evaluate_model(base_model, base_tokenizer, samples, "Base Model")

        # Free memory
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        base_results = [{'cobol': s['cobol'], 'reference': s['reference'], 'output': '',
                        'valid_syntax': False, 'syntax_error': 'skipped', 'bleu': 0, 'code_bleu': 0}
                       for s in samples]

    # Evaluate fine-tuned model
    ft_model, ft_tokenizer = load_finetuned_model(args.model, device_map, torch_dtype)
    finetuned_results = evaluate_model(ft_model, ft_tokenizer, samples, "Fine-tuned Model")

    # Print results
    if not args.finetuned_only:
        print_summary(base_results, finetuned_results)
        print_side_by_side(base_results, finetuned_results, args.examples)
    else:
        # Just show fine-tuned stats
        valid_pct = 100 * sum(1 for r in finetuned_results if r['valid_syntax']) / len(finetuned_results)
        avg_bleu = sum(r['bleu'] for r in finetuned_results) / len(finetuned_results)
        console.print(f"\n[bold]Fine-tuned Model Results:[/bold]")
        console.print(f"  Valid Syntax: {valid_pct:.1f}%")
        console.print(f"  Average BLEU: {avg_bleu:.4f}")

    # Save results
    save_results(base_results, finetuned_results, args.output)


if __name__ == "__main__":
    main()
