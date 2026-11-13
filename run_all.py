import csv
from pathlib import Path

from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer

from group_similarity import (
    CONFIG,
    load_texts,
    score_group,
    save_matrices,
)


def main():
    languages = {
        "tagalog": "tl",
        "chinese": "zh",
    }

    groups = {
        "human": "human_csv",
        "ai": "ai_csv",
    }

    metrics = [
        "bleu",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "bertscore_f1",
        "sbert_cosine",
    ]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Load models once for all 12 analyses
    print("Loading models...", flush=True)

    sbert_model = SentenceTransformer(
        CONFIG["sbert_model_name"]
    )

    bert_scorer = BERTScorer(
        model_type=CONFIG["bertscore_model_type"],
        batch_size=CONFIG["bertscore_batch_size"]
    )

    summary_rows = []

    for language, lang_code in languages.items():
        for group, folder in groups.items():

            # Store Q1, Q2, Q3 scores for each metric
            question_scores = {
                metric: [] for metric in metrics
            }

            for question in range(1, 4):
                csv_path = (
                    Path("data")
                    / folder
                    / f"{group}_{language}_q{question}.csv"
                )

                if not csv_path.exists():
                    raise FileNotFoundError(
                        f"Missing CSV: {csv_path}"
                    )

                print(f"\nRunning {csv_path}", flush=True)

                texts = load_texts(csv_path)

                averages, pair_scores = score_group(
                    texts,
                    lang_code,
                    bert_scorer,
                    sbert_model,
                )

                # Save the individual similarity matrices
                save_matrices(
                    pair_scores,
                    len(texts),
                    csv_path,
                )

                print("Question averages:")

                for metric in metrics:
                    score = averages[metric]
                    question_scores[metric].append(score)
                    print(f"  {metric}: {score:.4f}")

            # Calculate overall averages across Q1, Q2, Q3
            print(
                f"\nOVERALL AVERAGES: {group.upper()} "
                f"{language.upper()}"
            )
            print("-" * 45)

            for metric in metrics:
                scores = question_scores[metric]

                overall_average = sum(scores) / len(scores)

                calculation = (
                    f"({scores[0]:.6f} + "
                    f"{scores[1]:.6f} + "
                    f"{scores[2]:.6f}) / 3"
                )

                summary_rows.append({
                    "group": group,
                    "language": language,
                    "metric": metric,
                    "q1": f"{scores[0]:.6f}",
                    "q2": f"{scores[1]:.6f}",
                    "q3": f"{scores[2]:.6f}",
                    "calculation": calculation,
                    "overall_average": f"{overall_average:.6f}",
                })

                print(f"{metric}: {overall_average:.4f}")

    # Save overall summary CSV
    output_path = output_dir / "overall_similarity_averages.csv"

    with open(
        output_path, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "language",
                "metric",
                "q1",
                "q2",
                "q3",
                "calculation",
                "overall_average",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved overall averages to: {output_path}")
    print("Finished all 12 analyses!")


if __name__ == "__main__":
    main()
