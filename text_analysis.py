"""AI-vs-human translation evaluation pipeline.

Scores AI translations against human-written originals (same language) with
BLEU, ROUGE, BERTScore, and SBERT cosine similarity; rescales the embedding
metrics against a random-pair baseline; aggregates per question and overall;
and runs a cross-language significance test on the rescaled embedding
metrics only.
"""

import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from itertools import combinations

import numpy as np
import opencc
import pandas as pd
import sacrebleu
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("translation_eval")

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "languages": ["zh", "tl"],  # add "en" to also score English
    "input_csv": "data.csv",
    "output_dir": "outputs",
    "sbert_model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bertscore_model_type": "xlm-roberta-large",
    "bertscore_batch_size": 32,
    "sbert_batch_size": 32,
    "n_baseline_shuffles": 5,
    "baseline_seed": 42,
}

PINNED_PACKAGES = [
    "sacrebleu",
    "bert-score",
    "opencc-python-reimplemented",
    "rouge-score",
    "sentence-transformers",
    "scipy",
    "pandas",
    "numpy",
    "torch",
]

METRIC_COLS = [
    "bleu",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "bertscore_f1_raw",
    "bertscore_f1_rescaled",
    "sbert_cosine_raw",
    "sbert_cosine_rescaled",
]

# Only these are comparable across languages once rescaled; see
# run_cross_language_tests for why BLEU/ROUGE are excluded.
CROSS_LANG_METRICS = ["bertscore_f1_rescaled", "sbert_cosine_rescaled"]


def get_pinned_versions():
    out = {}
    for pkg in PINNED_PACKAGES:
        try:
            out[pkg] = pkg_version(pkg)
        except Exception:
            out[pkg] = "unknown"
    return out


# ============================================================
# DATA MODEL
# ============================================================
@dataclass
class Item:
    item_id: str
    question_id: str
    language: str
    human_text: str
    ai_text: str


def load_items(csv_path, languages):
    """Load item/question/language/author_type/text rows and pair human+ai
    texts by (item_id, language). Never pairs by row order."""
    df = pd.read_csv(csv_path, dtype=str)
    required_cols = {"item_id", "question_id", "language", "author_type", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    df = df[df["language"].isin(languages)]

    items = []
    for (item_id, language), group in df.groupby(["item_id", "language"]):
        humans = group[group["author_type"] == "human"]
        ais = group[group["author_type"] == "ai"]

        if len(humans) != 1 or len(ais) != 1:
            logger.warning(
                "Skipping item_id=%s language=%s: expected exactly one human "
                "and one ai text, found %d human, %d ai",
                item_id, language, len(humans), len(ais),
            )
            continue

        question_ids = group["question_id"].unique()
        if len(question_ids) != 1:
            logger.warning(
                "Skipping item_id=%s language=%s: inconsistent question_id %s",
                item_id, language, list(question_ids),
            )
            continue

        items.append(Item(
            item_id=str(item_id),
            question_id=str(question_ids[0]),
            language=language,
            human_text=humans.iloc[0]["text"],
            ai_text=ais.iloc[0]["text"],
        ))

    return items


# ============================================================
# TOKENIZATION (language-specific)
# ============================================================
_ZH_CONVERTER = opencc.OpenCC("t2s")


class CharTokenizer:
    """Character-level tokenizer for CJK text, used with rouge_score.

    rouge_score's DefaultTokenizer strips to [a-z0-9] word characters, which
    drops CJK characters entirely and silently returns ROUGE == 0 for zh.
    """

    def tokenize(self, text):
        return [ch for ch in text if not ch.isspace()]


_TL_WORD_RE = re.compile(r"[a-zñ]+(?:-[a-zñ]+)*")


class TagalogTokenizer:
    """Word tokenizer for Tagalog that keeps ñ and hyphenated words intact."""

    def tokenize(self, text):
        return _TL_WORD_RE.findall(text.lower())


def preprocess_text(text, lang):
    if lang == "zh":
        return _ZH_CONVERTER.convert(text)
    return text


def get_bleu_tokenize(lang):
    return {"zh": "zh", "tl": "13a", "en": "13a"}.get(lang, "13a")


def get_rouge_scorer(lang):
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    if lang == "zh":
        return rouge_scorer.RougeScorer(rouge_types, tokenizer=CharTokenizer())
    if lang == "tl":
        # The built-in Porter stemmer is English-only, so it must stay off;
        # a custom tokenizer already bypasses it, this is just documentation.
        return rouge_scorer.RougeScorer(rouge_types, use_stemmer=False, tokenizer=TagalogTokenizer())
    return rouge_scorer.RougeScorer(rouge_types)  # en: ROUGE default


# ============================================================
# SBERT ENCODING (with chunk + mean-pool for long texts)
# ============================================================
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。!?])\s+")


def split_sentences(text):
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return parts if parts else [text.strip()]


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def encode_texts(texts, model, batch_size):
    """Encode texts with the SBERT model, chunking+mean-pooling any text
    that would otherwise be silently truncated at max_seq_length."""
    max_len = model.max_seq_length
    token_counts = [len(model.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    chunked_flags = [tc > max_len for tc in token_counts]

    embeddings = [None] * len(texts)

    plain_idx = [i for i, c in enumerate(chunked_flags) if not c]
    if plain_idx:
        plain_embeds = model.encode([texts[i] for i in plain_idx], batch_size=batch_size, show_progress_bar=False)
        for i, emb in zip(plain_idx, plain_embeds):
            embeddings[i] = emb

    chunk_idx = [i for i, c in enumerate(chunked_flags) if c]
    for i in chunk_idx:
        sentences = split_sentences(texts[i])
        sent_embeds = model.encode(sentences, batch_size=batch_size, show_progress_bar=False)
        embeddings[i] = np.mean(sent_embeds, axis=0)

    if chunk_idx:
        logger.info(
            "SBERT: %d/%d texts exceeded max_seq_length=%d and were chunk-encoded + mean-pooled",
            len(chunk_idx), len(texts), max_len,
        )

    return np.array(embeddings), token_counts, chunked_flags


# ============================================================
# BERTSCORE
# ============================================================
def compute_bertscore_f1(scorer, candidates, references):
    _, _, f1 = scorer.score(candidates, references, batch_size=scorer.batch_size)
    return f1.numpy().tolist()


def write_bertscore_baseline_file(path, floor_value, num_layers):
    """Write floor_value in bert_score's custom baseline-file format
    (LAYER,P,R,F rows), so it can be reused via BERTScorer(baseline_path=...).
    """
    rows = [{"LAYER": layer, "P": floor_value, "R": floor_value, "F": floor_value} for layer in range(num_layers + 1)]
    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================
# PER-ITEM SCORING
# ============================================================
def score_items(items, lang, bert_scorer, sbert_model, sbert_batch_size):
    rouge_scorer_obj = get_rouge_scorer(lang)
    bleu_tokenize = get_bleu_tokenize(lang)

    candidates_pre = [preprocess_text(it.ai_text, lang) for it in items]
    references_pre = [preprocess_text(it.human_text, lang) for it in items]

    bleu_scores = [
        sacrebleu.sentence_bleu(c, [r], tokenize=bleu_tokenize).score / 100.0
        for c, r in zip(candidates_pre, references_pre)
    ]

    rouge1, rouge2, rougeL = [], [], []
    for c, r in zip(candidates_pre, references_pre):
        s = rouge_scorer_obj.score(r, c)
        rouge1.append(s["rouge1"].fmeasure)
        rouge2.append(s["rouge2"].fmeasure)
        rougeL.append(s["rougeL"].fmeasure)

    bertscore_f1 = compute_bertscore_f1(bert_scorer, candidates_pre, references_pre)

    n = len(items)
    all_texts = candidates_pre + references_pre
    embeddings, token_counts, chunked_flags = encode_texts(all_texts, sbert_model, sbert_batch_size)
    ai_embs, human_embs = embeddings[:n], embeddings[n:]
    ai_token_counts, human_token_counts = token_counts[:n], token_counts[n:]
    ai_chunked, human_chunked = chunked_flags[:n], chunked_flags[n:]

    sbert_cos = [cosine_sim(ai_embs[i], human_embs[i]) for i in range(n)]

    emb_cache = {
        "ai": {items[i].item_id: ai_embs[i] for i in range(n)},
        "human": {items[i].item_id: human_embs[i] for i in range(n)},
    }

    rows = []
    for i, it in enumerate(items):
        rows.append({
            "item_id": it.item_id,
            "question_id": it.question_id,
            "language": lang,
            "bleu": bleu_scores[i],
            "rouge1_f1": rouge1[i],
            "rouge2_f1": rouge2[i],
            "rougeL_f1": rougeL[i],
            "bertscore_f1_raw": bertscore_f1[i],
            "sbert_cosine_raw": sbert_cos[i],
            "ai_token_count": ai_token_counts[i],
            "human_token_count": human_token_counts[i],
            "ai_chunked": ai_chunked[i],
            "human_chunked": human_chunked[i],
        })

    return rows, emb_cache


# ============================================================
# RANDOM-PAIR BASELINE ("unrelated text" floor) + RESCALING
# ============================================================
def build_baseline(items, lang, bert_scorer, emb_cache, n_shuffles, seed):
    """Pair each AI translation with the human text of a different, randomly
    chosen item (a derangement), over n_shuffles fixed-seed shuffles, and
    average BERTScore F1 / SBERT cosine over all mismatched pairs."""
    n = len(items)
    if n < 2:
        logger.warning("Language=%s has fewer than 2 items; cannot build a baseline", lang)
        return None

    rng = random.Random(seed)
    perms = []
    for _ in range(n_shuffles):
        perm = list(range(n))
        while True:
            rng.shuffle(perm)
            if all(perm[i] != i for i in range(n)):
                break
        perms.append(perm)

    mismatched_candidates = []
    mismatched_references = []
    sbert_pairs = []
    for perm in perms:
        for i in range(n):
            mismatched_candidates.append(preprocess_text(items[i].ai_text, lang))
            mismatched_references.append(preprocess_text(items[perm[i]].human_text, lang))
            sbert_pairs.append((items[i].item_id, items[perm[i]].item_id))

    bert_f1s = compute_bertscore_f1(bert_scorer, mismatched_candidates, mismatched_references)
    sbert_coss = [
        cosine_sim(emb_cache["ai"][ai_id], emb_cache["human"][human_id])
        for ai_id, human_id in sbert_pairs
    ]

    return {
        "language": lang,
        "bertscore_f1_floor": float(np.mean(bert_f1s)),
        "sbert_cosine_floor": float(np.mean(sbert_coss)),
        "n_shuffles": n_shuffles,
        "n_pairs": len(bert_f1s),
    }


def rescale(raw, floor):
    denom = 1 - floor
    if denom == 0:
        return float("nan")
    return (raw - floor) / denom


def apply_rescaling(rows, baseline):
    floor_bert = baseline["bertscore_f1_floor"]
    floor_sbert = baseline["sbert_cosine_floor"]
    for row in rows:
        row["bertscore_f1_rescaled"] = rescale(row["bertscore_f1_raw"], floor_bert)
        row["sbert_cosine_rescaled"] = rescale(row["sbert_cosine_raw"], floor_sbert)
    return rows


# ============================================================
# AGGREGATION
# ============================================================
def aggregate_per_question(item_df):
    return (
        item_df.groupby(["language", "question_id"])[METRIC_COLS]
        .mean()
        .reset_index()
    )


def aggregate_overall(item_df, question_df):
    rows = []
    for lang, group in question_df.groupby("language"):
        row = {
            "language": lang,
            "n_items": int((item_df["language"] == lang).sum()),
            "n_questions": len(group),
        }
        for col in METRIC_COLS:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sd"] = group[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# STATISTICS
# ============================================================
def run_cross_language_tests(question_df, lang_a, lang_b):
    # BLEU and ROUGE are never compared across languages: zh is scored on
    # character tokens, tl/en on word tokens, so a BLEU/ROUGE score from one
    # language is not on the same scale as the other even though both happen
    # to be bounded in [0, 1]. BERTScore and SBERT are embedding-based and
    # become comparable once rescaled against each language's own
    # random-pair floor, so only those two are tested here.
    a = question_df[question_df["language"] == lang_a].set_index("question_id")
    b = question_df[question_df["language"] == lang_b].set_index("question_id")
    common_qs = sorted(set(a.index) & set(b.index))

    results = []
    for metric in CROSS_LANG_METRICS:
        if len(common_qs) < 1:
            results.append({
                "language_a": lang_a, "language_b": lang_b, "metric": metric,
                "n_questions": 0, "statistic": float("nan"), "p_value": float("nan"),
            })
            continue
        x = a.loc[common_qs, metric].to_numpy()
        y = b.loc[common_qs, metric].to_numpy()
        try:
            stat, p = wilcoxon(x, y)
        except ValueError as exc:
            logger.warning("Wilcoxon failed for %s (%s vs %s): %s", metric, lang_a, lang_b, exc)
            stat, p = float("nan"), float("nan")
        results.append({
            "language_a": lang_a, "language_b": lang_b, "metric": metric,
            "n_questions": len(common_qs), "statistic": stat, "p_value": p,
        })
    return results


# ============================================================
# PIPELINE ORCHESTRATION
# ============================================================
def run_pipeline(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    languages = CONFIG["languages"]

    items = load_items(csv_path, languages)
    items_by_lang = defaultdict(list)
    for it in items:
        items_by_lang[it.language].append(it)

    sbert_model = SentenceTransformer(CONFIG["sbert_model_name"])
    bert_scorer = BERTScorer(model_type=CONFIG["bertscore_model_type"], batch_size=CONFIG["bertscore_batch_size"])
    print(f"BERTScore hash: {bert_scorer.hash}")

    item_rows = []
    baseline_rows = []

    for lang in languages:
        lang_items = items_by_lang.get(lang, [])
        if not lang_items:
            logger.warning("No valid items found for language=%s; skipping", lang)
            continue

        rows, emb_cache = score_items(lang_items, lang, bert_scorer, sbert_model, CONFIG["sbert_batch_size"])

        baseline = build_baseline(
            lang_items, lang, bert_scorer, emb_cache,
            CONFIG["n_baseline_shuffles"], CONFIG["baseline_seed"],
        )
        if baseline is None:
            for row in rows:
                row["bertscore_f1_rescaled"] = float("nan")
                row["sbert_cosine_rescaled"] = float("nan")
        else:
            rows = apply_rescaling(rows, baseline)
            baseline_file = os.path.join(output_dir, f"bertscore_baseline_{lang}.tsv")
            write_bertscore_baseline_file(baseline_file, baseline["bertscore_f1_floor"], bert_scorer.num_layers)
            baseline_row = dict(baseline)
            baseline_row["bertscore_baseline_file"] = baseline_file
            baseline_rows.append(baseline_row)

        item_rows.extend(rows)

    item_df = pd.DataFrame(item_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    question_df = aggregate_per_question(item_df)
    overall_df = aggregate_overall(item_df, question_df)

    test_rows = []
    for lang_a, lang_b in combinations(languages, 2):
        test_rows.extend(run_cross_language_tests(question_df, lang_a, lang_b))
    tests_df = pd.DataFrame(test_rows)

    item_df.to_csv(os.path.join(output_dir, "item_scores.csv"), index=False)
    baseline_df.to_csv(os.path.join(output_dir, "baselines.csv"), index=False)
    question_df.to_csv(os.path.join(output_dir, "question_summary.csv"), index=False)
    overall_df.to_csv(os.path.join(output_dir, "overall_summary.csv"), index=False)
    tests_df.to_csv(os.path.join(output_dir, "tests.csv"), index=False)

    logger.info("Wrote outputs to %s", output_dir)
    return {
        "item_scores": item_df,
        "baselines": baseline_df,
        "question_summary": question_df,
        "overall_summary": overall_df,
        "tests": tests_df,
    }


# ============================================================
# SYNTHETIC TEST SET + SELF-TEST
# ============================================================
def build_synthetic_dataset():
    rows = []

    def add(item_id, question_id, language, human_text, ai_text):
        rows.append({"item_id": item_id, "question_id": question_id, "language": language,
                      "author_type": "human", "text": human_text})
        rows.append({"item_id": item_id, "question_id": question_id, "language": language,
                      "author_type": "ai", "text": ai_text})

    add("1", "q1", "zh", "我喜欢在周末看书和散步。", "我周末喜欢看书和散步。")
    add("2", "q1", "zh", "这个城市的天气今天非常好。", "今天这个城市的天气非常好。")
    add("3", "q2", "zh", "他正在学习如何编写程序。", "他在学习怎么编写程序。")

    add("1", "q1", "tl", "Mahal ko ang pagbasa ng libro at paglalakad sa Sabado.",
        "Mahal ko ang pagbasa at paglalakad sa Sabado.")
    add("2", "q1", "tl", "Ang panahon sa lungsod na ito ay maganda ngayon.",
        "Maganda ang panahon sa lungsod na ito ngayon.")
    add("3", "q2", "tl", "Siya ay natututo kung paano magsulat ng programa.",
        "Natututo siya kung paano sumulat ng programa.")

    return pd.DataFrame(rows)


def run_self_tests():
    """Regression check: the CJK-aware ROUGE tokenizer must not collapse to
    the default DefaultTokenizer, which drops CJK characters and scores 0."""
    scorer = get_rouge_scorer("zh")
    human = preprocess_text("今天天气很好。", "zh")
    ai = preprocess_text("今天的天气也很好。", "zh")
    result = scorer.score(human, ai)
    assert result["rouge1"].fmeasure > 0, (
        f"Expected ROUGE-1 > 0 for a zh pair sharing characters, got {result['rouge1'].fmeasure}"
    )
    logger.info("Self-test passed: zh ROUGE-1 F1 = %.3f for a shared-character pair", result["rouge1"].fmeasure)


def main():
    versions = get_pinned_versions()
    print("Pinned library versions:")
    for pkg, v in versions.items():
        print(f"  {pkg}=={v}")

    run_self_tests()

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    synthetic_csv = os.path.join(CONFIG["output_dir"], "synthetic_input.csv")
    build_synthetic_dataset().to_csv(synthetic_csv, index=False)

    logger.info("Running pipeline on synthetic test set: %s", synthetic_csv)
    run_pipeline(synthetic_csv, os.path.join(CONFIG["output_dir"], "synthetic"))

    if os.path.exists(CONFIG["input_csv"]):
        logger.info("Running pipeline on real input: %s", CONFIG["input_csv"])
        run_pipeline(CONFIG["input_csv"], os.path.join(CONFIG["output_dir"], "real"))
    else:
        logger.info("No real input CSV found at %s; synthetic run only.", CONFIG["input_csv"])


if __name__ == "__main__":
    main()
