#!/usr/bin/env python3
"""
Data Selection Script for COBOL->Python Fine-tuning
Uses sentence transformers + K-Means clustering + complexity scoring
to select the most representative and complex samples.
"""

import json
import re
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

# COBOL keywords that indicate complexity
COBOL_COMPLEXITY_KEYWORDS = [
    'PERFORM', 'EVALUATE', 'SEARCH', 'CALL', 'COPY', 'EXEC SQL', 'EXEC CICS',
    'COMPUTE', 'INSPECT', 'STRING', 'UNSTRING', 'REFERENCE MODIFICATION',
    'OCCURS', 'REDEFINES', 'RENAMES', 'CORRESPONDING', 'JUSTIFIED',
    'SYNCHRONIZED', 'USAGE COMP', 'USAGE PACKED', 'BINARY', 'POINTER',
    'PROCEDURE POINTER', 'FUNCTION', 'INTRINSIC', 'DECLARATIVES',
    'USE AFTER', 'AT END', 'INVALID KEY', 'ON SIZE ERROR', 'ON OVERFLOW',
    'NOT AT END', 'NOT INVALID KEY', 'FILE STATUS', 'SORT', 'MERGE',
    'START', 'DELETE', 'REWRITE', 'OPEN I-O', 'ACCESS DYNAMIC',
    'ORGANIZATION INDEXED', 'ORGANIZATION RELATIVE', 'ALTERNATE KEY',
    'DEPENDING ON', 'ASCENDING KEY', 'DESCENDING KEY', 'INDEXED BY'
]

# Python keywords/patterns that indicate complexity
PYTHON_COMPLEXITY_PATTERNS = [
    r'class\s+\w+', r'def\s+\w+', r'@\w+', r'lambda', r'yield',
    r'async\s+def', r'await\s+', r'try:', r'except\s+', r'finally:',
    r'with\s+', r'import\s+', r'from\s+\w+\s+import', r'\[.*for.*in.*\]',
    r'\{.*for.*in.*\}', r'\.join\(', r'\.format\(', r'f".*\{',
    r'isinstance\(', r'getattr\(', r'setattr\(', r'__\w+__',
    r'dataclass', r'property', r'staticmethod', r'classmethod'
]


def load_data(file_path: str) -> List[Dict]:
    """Load the COBOL-Python pairs from JSON file."""
    console.print(f"[bold blue]Loading data from {file_path}...[/bold blue]")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    console.print(f"[green]Loaded {len(data)} samples[/green]")
    return data


def calculate_complexity_score(cobol_code: str, python_code: str) -> float:
    """
    Calculate complexity score based on:
    1. Line count (normalized)
    2. COBOL keyword complexity
    3. Python pattern complexity
    """
    # Line count component
    cobol_lines = len(cobol_code.strip().split('\n'))
    python_lines = len(python_code.strip().split('\n'))
    line_score = (cobol_lines + python_lines) / 2

    # COBOL keyword complexity
    cobol_upper = cobol_code.upper()
    cobol_keyword_count = sum(1 for kw in COBOL_COMPLEXITY_KEYWORDS if kw in cobol_upper)

    # Python pattern complexity
    python_pattern_count = sum(1 for pattern in PYTHON_COMPLEXITY_PATTERNS
                                if re.search(pattern, python_code))

    # Weighted combination
    complexity = (
        line_score * 0.3 +           # 30% weight on line count
        cobol_keyword_count * 2.0 +   # 2 points per COBOL keyword
        python_pattern_count * 1.5    # 1.5 points per Python pattern
    )

    return complexity


def create_embeddings(data: List[Dict], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Create embeddings for each sample using sentence transformers."""
    console.print(f"[bold blue]Creating embeddings using {model_name}...[/bold blue]")

    model = SentenceTransformer(model_name)

    # Combine COBOL and Python code for embedding
    texts = []
    for item in tqdm(data, desc="Preparing texts"):
        # Use a combination of COBOL structure and Python implementation
        combined = f"COBOL: {item['cobol'][:500]} PYTHON: {item['python'][:500]}"
        texts.append(combined)

    console.print("[yellow]Encoding texts (this may take a moment)...[/yellow]")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    console.print(f"[green]Created embeddings with shape {embeddings.shape}[/green]")
    return embeddings


def cluster_data(embeddings: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """Cluster the data using K-Means."""
    console.print(f"[bold blue]Clustering data into {n_clusters} clusters...[/bold blue]")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Print cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    table = Table(title="Cluster Distribution")
    table.add_column("Cluster", style="cyan")
    table.add_column("Count", style="green")
    for cluster_id, count in zip(unique, counts):
        table.add_row(str(cluster_id), str(count))
    console.print(table)

    return cluster_labels


def select_samples(
    data: List[Dict],
    cluster_labels: np.ndarray,
    top_percentage: float = 0.4,
    train_ratio: float = 0.9
) -> Tuple[List[Dict], List[Dict]]:
    """
    Select top samples from each cluster based on complexity.
    Then split into train/validation sets.
    """
    console.print(f"[bold blue]Selecting top {int(top_percentage*100)}% most complex samples from each cluster...[/bold blue]")

    # Calculate complexity scores for all samples
    complexity_scores = []
    for item in tqdm(data, desc="Calculating complexity"):
        score = calculate_complexity_score(item['cobol'], item['python'])
        complexity_scores.append(score)

    complexity_scores = np.array(complexity_scores)

    # Select top samples from each cluster
    selected_indices = []
    n_clusters = len(np.unique(cluster_labels))

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_complexities = complexity_scores[cluster_mask]

        # Sort by complexity and take top percentage
        n_select = max(1, int(len(cluster_indices) * top_percentage))
        sorted_indices = np.argsort(cluster_complexities)[::-1][:n_select]
        selected_indices.extend(cluster_indices[sorted_indices])

    console.print(f"[green]Selected {len(selected_indices)} samples total[/green]")

    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(selected_indices)

    n_train = int(len(selected_indices) * train_ratio)
    train_indices = selected_indices[:n_train]
    val_indices = selected_indices[n_train:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]

    console.print(f"[green]Train samples: {len(train_data)}, Validation samples: {len(val_data)}[/green]")

    return train_data, val_data


def format_as_conversation(item: Dict) -> Dict:
    """Format a single item as a conversation with system/user/assistant messages."""
    system_message = (
        "You are an expert COBOL to Python translator. Your task is to convert "
        "COBOL programs into clean, idiomatic Python code that preserves the original "
        "program's functionality and logic. Use appropriate Python patterns like classes, "
        "dataclasses, context managers, and standard library modules."
    )

    user_message = f"Translate the following COBOL code to Python:\n\n```cobol\n{item['cobol']}\n```"

    assistant_message = f"```python\n{item['python']}\n```"

    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    }


def save_data(train_data: List[Dict], val_data: List[Dict], output_dir: str):
    """Save the processed data in conversational format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Format as conversations
    console.print("[bold blue]Formatting data as conversations...[/bold blue]")

    train_conversations = [format_as_conversation(item) for item in tqdm(train_data, desc="Formatting train")]
    val_conversations = [format_as_conversation(item) for item in tqdm(val_data, desc="Formatting val")]

    # Save as JSONL (one JSON object per line - better for streaming)
    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"

    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_conversations:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_conversations:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    console.print(f"[green]Saved train data to {train_file}[/green]")
    console.print(f"[green]Saved validation data to {val_file}[/green]")

    # Also save metadata
    metadata = {
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "total_samples": len(train_data) + len(val_data),
        "format": "conversational",
        "model_target": "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    console.print(f"[green]Saved metadata to {output_path / 'metadata.json'}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Select and prepare data for COBOL->Python fine-tuning")
    parser.add_argument("--input", type=str, default="data/cobol_python_pairs.json",
                        help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--n-clusters", type=int, default=10,
                        help="Number of K-Means clusters")
    parser.add_argument("--top-percentage", type=float, default=0.8,
                        help="Percentage of top complex samples to select from each cluster")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Train/validation split ratio")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model for embeddings")

    args = parser.parse_args()

    console.print("[bold green]" + "=" * 60)
    console.print("[bold green]COBOL->Python Data Selection Pipeline[/bold green]")
    console.print("[bold green]" + "=" * 60)

    # Load data
    data = load_data(args.input)

    # Create embeddings
    embeddings = create_embeddings(data, args.embedding_model)

    # Cluster
    cluster_labels = cluster_data(embeddings, args.n_clusters)

    # Select samples
    train_data, val_data = select_samples(
        data, cluster_labels,
        args.top_percentage,
        args.train_ratio
    )

    # Save
    save_data(train_data, val_data, args.output)

    console.print("\n[bold green]Data selection complete![/bold green]")
    console.print("[bold]Next step:[/bold] Run 'python train.py' to start fine-tuning")


if __name__ == "__main__":
    main()
