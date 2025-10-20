#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import re
from collections import defaultdict
from pathlib import Path

# ============================================================
# Helpers
# ============================================================

LETTERS = list("ABCDEFGHIJ")
base_data_dir = ("/projects/aclab/shayda/vqa_benchmarks")
def extract_answer_letter(s: str) -> str:
    """
    Extract a single letter A–J from a model's raw output.
    Falls back to "" if none is found (caller may random-guess).
    """
    if s is None:
        return ""
    s = str(s).strip()

    # Normalize common prefixes
    prefixes = [
        "The answer is", "The correct answer is", "The best answer is",
        "Answer:", "Option:", "### Final Answer:\n$$\\boxed", "the final answer is"
    ]
    for p in prefixes:
        s = s.replace(p, "")

    # If very long and no letter hint, bail early
    if len(s.split()) > 16 and not re.search(r"[A-J]", s):
        return ""

    m = re.search(r"[A-J]", s)
    return m[0] if m else ""

def _sorted_option_letters(doc) -> list:
    """
    Return option letters present in doc["options"] in A..J order.
    """
    opts = doc.get("options") or {}
    letters_present = [L for L in LETTERS if L in opts]
    return letters_present

# ============================================================
# lmms-eval style hooks (text & visual adapters are minimal)
# ============================================================

# TODO: fill this in for path from dataset to video_id for each of the 5 benchmarks
def motion_analysis_bench_doc_to_visual(doc):
    """
    If you want to feed a video path to a VLM, implement your pathing here.
    For now, we return an empty list to indicate 'no visual file provided'.
    """
    video_path = ""
    video_id = ""
    benchmark = doc.get("benchmark")
    base_data_dir = ("/projects/aclab/shayda/vqa_benchmarks")
    if benchmark == 'egoschema':
        video_id = doc.get("question_id") #for egoschema only question id maps to vidoe id for pathing purposes 
    else:
        video_id = doc.get("video_id")
    
    video_path = os.path.join(base_data_dir, f"{benchmark}/data/{video_id}.mp4")
    return [video_path]  # or return [<path>] if available

def _format_options_for_prompt(opts: dict) -> str:
    """
    Render options into a string:
      A. text
      B. text
    """
    letters = [L for L in LETTERS if L in opts]
    return "\n".join(f"{L}. {opts[L]}" for L in letters)

def motion_analysis_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Build the textual prompt. Keeps things simple and dataset-agnostic.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt  = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc.get("question", "")
    options  = doc.get("options", {}) or {}
    options_text = _format_options_for_prompt(options)

    input_text = f"{pre_prompt}{question}\n{options_text}\n{post_prompt}"
    return input_text

def motion_analysis_bench_doc_to_choice(doc):
    """
    Return the list of valid choice letters for this sample, e.g. ["A","B","C","D","E"]
    based on the actual options present.
    """
    return _sorted_option_letters(doc)

def motion_analysis_bench_doc_to_target(doc):
    """
    Return the index (0-based) of the correct answer within the available options list.
    """
    letters = _sorted_option_letters(doc)
    ans = (doc.get("answer") or "").strip()
    try:
        return letters.index(ans)
    except ValueError:
        # If missing or mismatch, default to -1 to avoid crashing (some harnesses expect int).
        # You can also raise here if you want strictness.
        return -1

# ============================================================
# Evaluation (per-sample processing + aggregation)
# ============================================================

def motion_analysis_bench_process_results(doc, results):
    """
    Convert a model's raw output for a single item into a normalized dict
    that includes correctness and attributes we want to aggregate on.
    """
    # Normalize raw output
    pred_raw = results[0] if isinstance(results, list) else results
    pred = extract_answer_letter(pred_raw)

    # If empty, random guess from available options A..(present)
    letters_present = _sorted_option_letters(doc)
    if not letters_present:
        # Degenerate case: no options—treat as incorrect
        pred = ""
    elif pred not in letters_present:
        pred = random.choice(letters_present)

    gold = (doc.get("answer") or "").strip()
    correct = (pred == gold) if gold in letters_present else False

    payload = {
        # identifiers
        "id": doc.get("video_id"),
        "question_id": doc.get("question_id"),

        # attributes for breakdowns
        "question_type": doc.get("question_type", "UNKNOWN"),
        "duration_bucket": doc.get("duration_bucket", "UNKNOWN"),
        "motion_number": doc.get("motion_number", "UNKNOWN"),
        "timestamps_in_question": doc.get("timestamps_in_question", "UNKNOWN"),
        "object_involved": doc.get("object_involved", "UNKNOWN"),
        "egocentric": doc.get("egocentric", "UNKNOWN"),

        # answers
        "pred_answer": pred,
        "answer": gold,
        "correct": bool(correct),

        # keep raw for debugging
        "raw_output": pred_raw,
    }

    # lmms-eval expects a dict of metric_name -> payload
    return {"motion_analysis_bench_acc": payload}

def _unwrap_motion_result(rec):
    """
    Handle either flat dicts or {'motion_analysis_bench_acc': {...}}.
    """
    if isinstance(rec, dict) and "motion_analysis_bench_acc" in rec:
        return rec["motion_analysis_bench_acc"]
    return rec

def _aggregate_by_field(flat_results, field):
    bucket = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in flat_results:
        key = r.get(field, "UNKNOWN")
        is_correct = 1 if r.get("correct", False) else 0
        bucket[key]["correct"] += is_correct
        bucket[key]["total"] += 1

    summary = {}
    for k, v in bucket.items():
        acc = (v["correct"] / v["total"]) if v["total"] else 0.0
        summary[k] = {"num": v["total"], "acc": round(acc * 100, 2)}
    return summary

def motion_analysis_bench_aggregate_results(results):
    """
    Aggregate a list of per-sample results into:
      - overall accuracy
      - per-question_type
      - per-duration_bucket
      - per-motion_number
      - per-timestamps_in_question
      - per-object_involved
      - per-egocentric
    """
    flat = [_unwrap_motion_result(r) for r in results]

    # Overall
    total = len(flat)
    total_correct = sum(1 for r in flat if r.get("correct"))
    overall_acc = round((total_correct / total) * 100, 2) if total else 0.0

    fields = [
        "question_type",
        "duration_bucket",
        "motion_number",
        "timestamps_in_question",
        "object_involved",
        "egocentric",
    ]
    per_attr = {f: _aggregate_by_field(flat, f) for f in fields}

    # Pretty print
    print("\nMotion Analysis Bench — Evaluation Results")
    print(f"Overall Accuracy: {overall_acc}% (samples: {total})")

    for f in fields:
        print(f"\nStatistics by {f}:")
        for k, v in per_attr[f].items():
            print(f"{k}: {v['acc']}% (samples: {v['num']})")

    # Structured return for programmatic use
    return {
        "overall_acc": overall_acc,
        "by_question_type": per_attr["question_type"],
        "by_duration_bucket": per_attr["duration_bucket"],
        "by_motion_number": per_attr["motion_number"],
        "by_timestamps_in_question": per_attr["timestamps_in_question"],
        "by_object_involved": per_attr["object_involved"],
        "by_egocentric": per_attr["egocentric"],
    }

# ============================================================
# Optional: tiny self-test (run manually if needed)
# ============================================================
if __name__ == "__main__":
    # Minimal sanity check with one item
    sample = {
        "global_id": "global_000007",
        "benchmark": "egoschema",
        "video_id": "1ZdZ8aUcBNzndj135bqFrxb9L816EMGp1",
        "question_id": "0074f737-11cb-497d-8d07-77c3a8127391",
        "question": "Taking into account all the actions performed by c, what can you deduce?",
        "options": {
            "A": "C is cooking.",
            "B": "C is doing laundry.",
            "C": "C is cleaning the kitchen.",
            "D": "C is cleaning dishes.",
            "E": "C is cleaning the bathroom."
        },
        "answer": "D",
        "question_type": "Reasoning",
        "video_duration": None,
        "duration_bucket": "medium",
        "motion_number": "Single",
        "timestamps_in_question": "N",
        "object_involved": "Y",
        "egocentric": "Y",
    }

    # Fake model output resembling a chain-of-thought followed by an answer letter
    fake_output = "After analysis, the correct choice is D."

    one = motion_analysis_bench_process_results(sample, fake_output)
    agg = motion_analysis_bench_aggregate_results([one])
    print("\nReturn dict:\n", agg)
