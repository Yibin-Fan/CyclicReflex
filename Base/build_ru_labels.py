import argparse
import json
import os
from typing import Any, Dict, List

import torch


def load_pt_outputs(path: str) -> List[Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "outputs" in obj:
        outputs = obj["outputs"]
    elif isinstance(obj, list):
        outputs = obj
    else:
        raise ValueError(f"Unrecognized .pt structure in {path}")

    if not isinstance(outputs, list):
        raise ValueError(f"'outputs' is not a list in {path}")

    return outputs


def normalize_bool(x: Any) -> int:
    return int(bool(x))


def safe_get(sample: Dict[str, Any], key: str, default=None):
    return sample.get(key, default) if isinstance(sample, dict) else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument("--tip_mild", type=str, required=True)
    parser.add_argument("--tip_strong", type=str, required=True)
    parser.add_argument("--cyclic", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--output_strong_jsonl", type=str, required=True)
    args = parser.parse_args()

    original = load_pt_outputs(args.original)
    tip_mild = load_pt_outputs(args.tip_mild)
    tip_strong = load_pt_outputs(args.tip_strong)
    cyclic = load_pt_outputs(args.cyclic)

    n = len(original)
    assert len(tip_mild) == n, "tip_mild length mismatch"
    assert len(tip_strong) == n, "tip_strong length mismatch"
    assert len(cyclic) == n, "cyclic length mismatch"

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    rows = []
    strong_rows = []

    stats = {
        "n_total": 0,
        "ru_pos": 0,
        "ru_zero": 0,
        "ru_neg": 0,
    }

    for i in range(n):
        s0 = original[i]
        s1 = tip_mild[i]
        s2 = tip_strong[i]
        s3 = cyclic[i]

        # 基础一致性检查
        q0 = safe_get(s0, "question")
        q1 = safe_get(s1, "question")
        q2 = safe_get(s2, "question")
        q3 = safe_get(s3, "question")

        if not (q0 == q1 == q2 == q3):
            raise ValueError(
                f"Question mismatch at index {i}\n"
                f"original={q0}\n"
                f"tip_mild={q1}\n"
                f"tip_strong={q2}\n"
                f"cyclic={q3}"
            )

        g0 = safe_get(s0, "gold_answer")
        g1 = safe_get(s1, "gold_answer")
        g2 = safe_get(s2, "gold_answer")
        g3 = safe_get(s3, "gold_answer")

        if not (g0 == g1 == g2 == g3):
            raise ValueError(
                f"Gold answer mismatch at index {i}\n"
                f"original={g0}\n"
                f"tip_mild={g1}\n"
                f"tip_strong={g2}\n"
                f"cyclic={g3}"
            )

        original_correct = normalize_bool(safe_get(s0, "correct", 0))
        tip_mild_correct = normalize_bool(safe_get(s1, "correct", 0))
        tip_strong_correct = normalize_bool(safe_get(s2, "correct", 0))
        cyclic_correct = normalize_bool(safe_get(s3, "correct", 0))

        conservative_scores = {
            "original": original_correct,
            "tip_mild": tip_mild_correct,
            "tip_strong": tip_strong_correct,
        }

        conservative_best_policy = max(
            conservative_scores,
            key=lambda k: conservative_scores[k]
        )
        conservative_best = conservative_scores[conservative_best_policy]

        boost_best_policy = "cyclic"
        boost_best = cyclic_correct

        ru = boost_best - conservative_best
        # 映射成三分类 boost-worthiness
        # +1: boost-helpful
        #  0: neutral
        # -1: boost-harmful
        boost_label = ru

        sample_id = f"{args.dataset}_{i:04d}"

        row = {
            "sample_id": sample_id,
            "dataset": args.dataset,
            "index": i,
            "question": q0,
            "gold_answer": g0,
            "difficulty_level": safe_get(s0, "difficulty_level", None),
            "ru": ru,
            "boost_label": boost_label,
            "conservative_best": conservative_best,
            "boost_best": boost_best,
            "best_conservative_policy": conservative_best_policy,
            "best_boost_policy": boost_best_policy,
            "scores": {
                "original": original_correct,
                "tip_mild": tip_mild_correct,
                "tip_strong": tip_strong_correct,
                "cyclic": cyclic_correct,
            },
            "predicted_answers": {
                "original": safe_get(s0, "predicted_answer"),
                "tip_mild": safe_get(s1, "predicted_answer"),
                "tip_strong": safe_get(s2, "predicted_answer"),
                "cyclic": safe_get(s3, "predicted_answer"),
            },
            "generation_lengths": {
                "original": safe_get(s0, "generation_length"),
                "tip_mild": safe_get(s1, "generation_length"),
                "tip_strong": safe_get(s2, "generation_length"),
                "cyclic": safe_get(s3, "generation_length"),
            }
        }

        rows.append(row)

        stats["n_total"] += 1
        if ru == 1:
            stats["ru_pos"] += 1
            strong_rows.append(row)
        elif ru == 0:
            stats["ru_zero"] += 1
        elif ru == -1:
            stats["ru_neg"] += 1
            strong_rows.append(row)
        else:
            raise ValueError(f"Unexpected RU value {ru} at index {i}")

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.output_strong_jsonl, "w", encoding="utf-8") as f:
        for row in strong_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("Finished building RU labels")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"All labels saved to: {args.output_jsonl}")
    print(f"Strong-only labels saved to: {args.output_strong_jsonl}")
    print("=" * 80)


if __name__ == "__main__":
    main()