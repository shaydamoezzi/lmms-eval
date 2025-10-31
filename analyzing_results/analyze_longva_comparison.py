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


def collect_results(files: List[str], qid2gid: Dict[str, str], gid2motion: Dict[str, bool]) -> List[Tuple[bool, bool]]:
    out: List[Tuple[bool, bool]] = []
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
                if qid is None or correct is None:
                    continue
                gid = qid2gid.get(qid)
                if gid is None:
                    continue
                has_motion = gid2motion.get(gid)
                if has_motion is None:
                    continue
                out.append((bool(correct), bool(has_motion)))
    return out


def collect_results_with_duration(files: List[str], qid2gid: Dict[str, str], gid2motion: Dict[str, bool]) -> List[Tuple[bool, bool, str]]:
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


def compute_accuracy(pairs: List[Tuple[bool, bool]]) -> Dict[str, float]:
    total = len(pairs)
    motion_pairs = [c for c in pairs if c[1] is True]
    non_motion_pairs = [c for c in pairs if c[1] is False]

    def acc(lst: List[Tuple[bool, bool]]):
        if not lst:
            return None, 0, 0
        num_correct = sum(1 for c, _ in lst if c)
        return num_correct / len(lst), num_correct, len(lst) - num_correct

    overall_acc, _, _ = acc(pairs) if total else (None, 0, 0)
    motion_acc, motion_correct, motion_incorrect = acc(motion_pairs)
    non_motion_acc, non_motion_correct, non_motion_incorrect = acc(non_motion_pairs)

    return {
        'overall': overall_acc if overall_acc is not None else float('nan'),
        'motion': motion_acc if motion_acc is not None else float('nan'),
        'non_motion': non_motion_acc if non_motion_acc is not None else float('nan'),
        'motion_correct': motion_correct,
        'motion_incorrect': motion_incorrect,
        'non_motion_correct': non_motion_correct,
        'non_motion_incorrect': non_motion_incorrect,
    }


def main():
    qa_jsonl = 'compiled_qa_data/qa.jsonl' if os.path.exists('compiled_qa_data/qa.jsonl') else 'qa.jsonl'
    qa_labeled = 'compiled_qa_data/qa_data_labeled.json' if os.path.exists('compiled_qa_data/qa_data_labeled.json') else 'qa_data_labeled.json'

    qid2gid = load_question_to_global_id(qa_jsonl)
    gid2motion = load_global_id_to_motion(qa_labeled)

    # Discover LongVA result files
    nextqa_files = sorted(glob.glob('output/longva-nextqa-*/**/*samples_motion_analysis_bench.jsonl'))
    egoschema_files = sorted(glob.glob('output/longva-egoschema-*/**/*samples_motion_analysis_bench.jsonl'))
    lvb_files = sorted(glob.glob('output/longva-longvideobench-*/**/*samples_motion_analysis_bench.jsonl'))
    lv_files = sorted(glob.glob('output/longva-lvbench-*/**/*samples_motion_analysis_bench.jsonl'))

    nq_triplets = [(c, m, 'short') for (c, m) in collect_results(nextqa_files, qid2gid, gid2motion)]
    ego_triplets = [(c, m, 'medium') for (c, m) in collect_results(egoschema_files, qid2gid, gid2motion)]
    lvb_triplets = collect_results_with_duration(lvb_files, qid2gid, gid2motion)
    lv_triplets = [(c, m, 'long') for (c, m) in collect_results(lv_files, qid2gid, gid2motion)]

    # Ordered entries: include points if present
    entries = []
    if nq_triplets:
        entries.append(('Short - NextQA', nq_triplets))
    if any(d == 'short' for _, _, d in lvb_triplets):
        entries.append(('Short - LVB-S', [(c, m, d) for (c, m, d) in lvb_triplets if d == 'short']))
    if ego_triplets:
        entries.append(('Medium - EgoSchema', ego_triplets))
    if any(d == 'medium' for _, _, d in lvb_triplets):
        entries.append(('Medium - LVB-M', [(c, m, d) for (c, m, d) in lvb_triplets if d == 'medium']))
    if any(d == 'long' for _, _, d in lvb_triplets):
        entries.append(('Long - LVB-L', [(c, m, d) for (c, m, d) in lvb_triplets if d == 'long']))
    if lv_triplets:
        entries.append(('Long - LVBench', lv_triplets))

    labels = [name for name, _ in entries]
    motion_vals = []
    non_motion_vals = []

    for _, triplets in entries:
        pairs = [(c, m) for (c, m, _) in triplets]
        stats = compute_accuracy(pairs)
        motion_vals.append(stats['motion'])
        non_motion_vals.append(stats['non_motion'])

    x = np.arange(len(labels))

    out_dir = os.path.join('longva-7b', 'comparison_results')
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    mv = np.array(motion_vals, dtype=float)
    nmv = np.array(non_motion_vals, dtype=float)
    mask_m = np.isfinite(mv)
    mask_nm = np.isfinite(nmv)

    plt.plot(x[mask_m], mv[mask_m], marker='o', linewidth=2.5, color='#4C78A8', label='Motion')
    plt.plot(x[mask_nm], nmv[mask_nm], marker='s', linewidth=2.5, color='#F58518', label='Non-Motion')

    plt.xticks(x, labels, rotation=20, ha='right')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('LongVA-7B: QA Accuracy vs Video Duration (Motion vs Non-Motion)')
    plt.grid(alpha=0.25, linestyle='--')
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'motion_nonmotion_accuracy_by_duration.png')
    plt.savefig(out_path, dpi=300)
    print('Saved', out_path)


if __name__ == '__main__':
    main()
