import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import sacrebleu
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from pathlib import Path

from text_analysis import (
    CONFIG,
    preprocess_text,
    get_bleu_tokenize,
    get_rouge_scorer,
    compute_bertscore_f1,
    encode_texts,
    cosine_sim,
)


def load_texts(csv_path):
    """Load all responses from a CSV with a 'text' column."""
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("CSV must have a 'text' column.")

    texts = df["text"].dropna().astype(str).tolist()
    texts = [text for text in texts if text.strip()]

    if len(texts) < 2:
        raise ValueError("Need at least 2 texts to compare.")

    return texts



def score_group(texts, lang, bert_scorer, sbert_model):
    """Calculate pairwise similarity and averages within one group."""

    texts = [preprocess_text(text, lang) for text in texts]
    pairs = list(combinations(range(len(texts)), 2))

    pair_scores = {
        "bleu": {},
        "rouge1_f1": {},
        "rouge2_f1": {},
        "rougeL_f1": {},
        "bertscore_f1": {},
        "sbert_cosine": {},
    }

    rouge = get_rouge_scorer(lang)

    for i, j in pairs:
        text1 = texts[i]
        text2 = texts[j]

        # BLEU: average both directions
        bleu_forward = sacrebleu.sentence_bleu(
            text1, [text2], tokenize=get_bleu_tokenize(lang)
        ).score / 100.0

        bleu_backward = sacrebleu.sentence_bleu(
            text2, [text1], tokenize=get_bleu_tokenize(lang)
        ).score / 100.0

        pair_scores["bleu"][(i, j)] = (
            bleu_forward + bleu_backward
        ) / 2

        # ROUGE
        scores = rouge.score(text1, text2)

        pair_scores["rouge1_f1"][(i, j)] = scores["rouge1"].fmeasure
        pair_scores["rouge2_f1"][(i, j)] = scores["rouge2"].fmeasure
        pair_scores["rougeL_f1"][(i, j)] = scores["rougeL"].fmeasure

    # BERTScore: average both directions
    candidates = [texts[i] for i, j in pairs]
    references = [texts[j] for i, j in pairs]

    bert_forward = compute_bertscore_f1(
        bert_scorer, candidates, references
    )
    bert_backward = compute_bertscore_f1(
        bert_scorer, references, candidates
    )

    for (i, j), a, b in zip(pairs, bert_forward, bert_backward):
        pair_scores["bertscore_f1"][(i, j)] = float((a + b) / 2)

    # SBERT
    embeddings, _, _ = encode_texts(
        texts, sbert_model, CONFIG["sbert_batch_size"]
    )

    for i, j in pairs:
        pair_scores["sbert_cosine"][(i, j)] = float(
            cosine_sim(embeddings[i], embeddings[j])
        )

    # Calculate averages
    averages = {
        metric: float(np.mean(list(scores.values())))
        for metric, scores in pair_scores.items()
    }

    return averages, pair_scores

def save_matrices(pair_scores, num_texts, csv_path):
    """Save all similarity matrices into one CSV."""

    csv_name = Path(csv_path).stem

    # Automatically label Human or AI responses
    if csv_name.lower().startswith("ai_"):
        labels = [f"AI {i+1}" for i in range(num_texts)]
    else:
        labels = [f"Person{i+1}" for i in range(num_texts)]

    output_rows = []

    for metric, scores in pair_scores.items():
        matrix = np.full((num_texts, num_texts), np.nan)

        for (i, j), score in scores.items():
            matrix[i, j] = score
            matrix[j, i] = score

        # Add a heading for this metric
        output_rows.append([metric.upper()])

        # Add column labels
        output_rows.append([""] + labels)

        # Add each row of the matrix
        for i in range(num_texts):
            row = [labels[i]]

            for j in range(num_texts):
                if i == j:
                    row.append("")
                else:
                    row.append(f"{matrix[i, j]:.4f}")

            output_rows.append(row)

        # Add average
        average = np.nanmean(matrix)
        output_rows.append(["Average", f"{average:.4f}"])

        # Blank row between metrics
        output_rows.append([])

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{csv_name}_similarity.csv"

    pd.DataFrame(output_rows).to_csv(
        output_path,
        index=False,
        header=False
    )

    print(f"\nSaved similarity matrices to: {output_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="CSV file to score")
    parser.add_argument(
        "--lang",
        required=True,
        choices=["tl", "zh", "en"],
        help="Language of the responses"
    )

    args = parser.parse_args()

    texts = load_texts(args.csv_path)

    print(f"Loaded {len(texts)} responses")
    print(f"Comparing {len(texts) * (len(texts) - 1) // 2} pairs")

    # Load the same models used by the original code
    sbert_model = SentenceTransformer(
        CONFIG["sbert_model_name"]
    )

    bert_scorer = BERTScorer(
        model_type=CONFIG["bertscore_model_type"],
        batch_size=CONFIG["bertscore_batch_size"]
    )

    averages, pair_scores = score_group(
        texts, args.lang, bert_scorer, sbert_model
    )

    print(f"\nAverage similarity for {args.csv_path}")
    print("-" * 40)

    for metric, score in averages.items():
        print(f"{metric}: {score:.4f}")

    save_matrices(pair_scores, len(texts), args.csv_path)


if __name__ == "__main__":
    main()
