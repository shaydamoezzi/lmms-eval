import json
import os
import glob
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_question_to_global_id(qa_jsonl_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(qa_jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Reading qa.jsonl'):
            d = json.loads(line)
            mapping[d['question_id']] = d['global_id']
    return mapping


def load_global_id_to_motion(qa_labeled_path: str) -> Dict[str, bool]:
    mapping: Dict[str, bool] = {}
    with open(qa_labeled_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in tqdm(data, desc='Reading qa_data_labeled.json'):
            gid = item.get('global_id')
            if gid is not None and 'has_motion' in item:
                mapping[gid] = bool(item['has_motion'])
    return mapping


def collect_results(files: List[str], qid2gid: Dict[str, str], gid2motion: Dict[str, bool]) -> List[Tuple[bool, bool, str]]:
    # returns list of (correct, has_motion, duration_bucket)
    out: List[Tuple[bool, bool, str]] = []
    for fp in files:
        if not os.path.exists(fp):
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f'Processing {os.path.basename(fp)}', leave=False):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                bench = d.get('motion_analysis_bench_acc') or {}
                qid = bench.get('question_id')
                correct = bench.get('correct')
                duration_bucket = (bench.get('duration_bucket') or '').lower()
                if qid is None or correct is None or not duration_bucket:
                    continue
                gid = qid2gid.get(qid)
                if gid is None:
                    continue
                has_motion = gid2motion.get(gid)
                if has_motion is None:
                    continue
                out.append((bool(correct), bool(has_motion), duration_bucket))
    return out


def main():
    # Prefer compiled_qa_data
    qa_jsonl = 'compiled_qa_data/qa.jsonl' if os.path.exists('compiled_qa_data/qa.jsonl') else 'qa.jsonl'
    qa_labeled = 'compiled_qa_data/qa_data_labeled.json' if os.path.exists('compiled_qa_data/qa_data_labeled.json') else 'qa_data_labeled.json'

    qid2gid = load_question_to_global_id(qa_jsonl)
    gid2motion = load_global_id_to_motion(qa_labeled)

    # Auto-discover LongVA LongVideoBench results
    files = sorted(glob.glob('output/longva-longvideobench-*/**/*samples_motion_analysis_bench.jsonl'))

    triplets = collect_results(files, qid2gid, gid2motion)

    # Aggregate by duration and motion
    durations = ['short', 'medium', 'long']
    counts = {d: {'motion': {'correct': 0, 'total': 0}, 'non_motion': {'correct': 0, 'total': 0}} for d in durations}

    for correct, has_motion, bucket in triplets:
        if bucket not in counts:
            continue
        key = 'motion' if has_motion else 'non_motion'
        counts[bucket][key]['total'] += 1
        if correct:
            counts[bucket][key]['correct'] += 1

    # Build arrays for plotting
    present = [d for d in durations if counts[d]['motion']['total'] + counts[d]['non_motion']['total'] > 0]
    x = np.arange(len(present))
    motion_vals = []
    non_motion_vals = []
    motion_correct = []
    motion_total = []
    non_motion_correct = []
    non_motion_total = []
    for d in present:
        mc = counts[d]['motion']['correct']; mt = counts[d]['motion']['total']
        nmc = counts[d]['non_motion']['correct']; nmt = counts[d]['non_motion']['total']
        motion_vals.append(mc / mt if mt else np.nan)
        non_motion_vals.append(nmc / nmt if nmt else np.nan)
        motion_correct.append(mc); motion_total.append(mt)
        non_motion_correct.append(nmc); non_motion_total.append(nmt)

    out_dir = os.path.join('longva-7b', 'longvideobench_results')
    os.makedirs(out_dir, exist_ok=True)

    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, motion_vals, width, label='Motion', color='#4C78A8')
    rects2 = ax.bar(x + width/2, non_motion_vals, width, label='Non-Motion', color='#F58518')

    ax.set_ylabel('Accuracy')
    ax.set_title('LongVA-7B LongVideoBench: Accuracy by Duration and Motion Type')
    ax.set_xticks(x, [d.capitalize() for d in present])
    ax.set_ylim(0, 1)
    ax.legend()

    def autolabel(rects, correct_counts, total_counts):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            total = total_counts[idx]
            correct = correct_counts[idx]
            if total == 0 or np.isnan(height):
                label = 'N/A'; y = 0.02
            else:
                label = f'{correct}/{total}'; y = height + 0.02
            ax.text(rect.get_x() + rect.get_width()/2., y, label,
                    ha='center', va='bottom', fontsize=9)

    autolabel(rects1, motion_correct, motion_total)
    autolabel(rects2, non_motion_correct, non_motion_total)

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'duration_motion_accuracy_longvideobench_longva7b.png')
    plt.savefig(out_path)
    print('Saved', out_path)


if __name__ == '__main__':
    main()
