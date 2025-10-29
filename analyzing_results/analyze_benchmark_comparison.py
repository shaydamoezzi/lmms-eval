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
    # Prefer compiled_qa_data
    qa_jsonl = 'compiled_qa_data/qa.jsonl' if os.path.exists('compiled_qa_data/qa.jsonl') else 'qa.jsonl'
    qa_labeled = 'compiled_qa_data/qa_data_labeled.json' if os.path.exists('compiled_qa_data/qa_data_labeled.json') else 'qa_data_labeled.json'

    qid2gid = load_question_to_global_id(qa_jsonl)
    gid2motion = load_global_id_to_motion(qa_labeled)

    # Auto-discover files for each benchmark
    lvbench_files = sorted(glob.glob('output/internvl-lvbench-*/OpenGVLab__InternVL2-8B/*.jsonl'))
    longvideobench_files = sorted(glob.glob('output/internvl-longvideobench-*/OpenGVLab__InternVL2-8B/*.jsonl'))
    nextqa_files = sorted(glob.glob('output/internvl-nextqa-*/OpenGVLab__InternVL2-8B/*.jsonl'))
    egoschema_files = sorted(glob.glob('output/internvl-egoschema-*/OpenGVLab__InternVL2-8B/*.jsonl'))

    print('Collecting LVBench...')
    lv_pairs = collect_results(lvbench_files, qid2gid, gid2motion)
    print('Collecting LongVideoBench...')
    lvb_pairs = collect_results(longvideobench_files, qid2gid, gid2motion)
    print('Collecting NextQA...')
    nq_pairs = collect_results(nextqa_files, qid2gid, gid2motion)
    print('Collecting EgoSchema...')
    ego_pairs = collect_results(egoschema_files, qid2gid, gid2motion)

    lv_stats = compute_accuracy(lv_pairs)
    lvb_stats = compute_accuracy(lvb_pairs)
    nq_stats = compute_accuracy(nq_pairs)
    ego_stats = compute_accuracy(ego_pairs)

    groups = ['Short (NextQA)', 'Medium (EgoSchema)', 'Long (LVBench)', 'Long (LongVideoBench)']
    motion_vals = [nq_stats['motion'], ego_stats['motion'], lv_stats['motion'], lvb_stats['motion']]
    non_motion_vals = [nq_stats['non_motion'], ego_stats['non_motion'], lv_stats['non_motion'], lvb_stats['non_motion']]

    # counts for labels
    motion_correct = [nq_stats['motion_correct'], ego_stats['motion_correct'], lv_stats['motion_correct'], lvb_stats['motion_correct']]
    motion_total = [nq_stats['motion_correct'] + nq_stats['motion_incorrect'],
                    ego_stats['motion_correct'] + ego_stats['motion_incorrect'],
                    lv_stats['motion_correct'] + lv_stats['motion_incorrect'],
                    lvb_stats['motion_correct'] + lvb_stats['motion_incorrect']]

    non_motion_correct = [nq_stats['non_motion_correct'], ego_stats['non_motion_correct'], lv_stats['non_motion_correct'], lvb_stats['non_motion_correct']]
    non_motion_total = [nq_stats['non_motion_correct'] + nq_stats['non_motion_incorrect'],
                        ego_stats['non_motion_correct'] + ego_stats['non_motion_incorrect'],
                        lv_stats['non_motion_correct'] + lv_stats['non_motion_incorrect'],
                        lvb_stats['non_motion_correct'] + lvb_stats['non_motion_incorrect']]

    out_dir = os.path.join('internvl-8b', 'comparison_results')
    os.makedirs(out_dir, exist_ok=True)

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, motion_vals, width, label='Motion', color='#4C78A8')
    rects2 = ax.bar(x + width/2, non_motion_vals, width, label='Non-Motion', color='#F58518')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Duration Group and Motion Type')
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1)
    ax.legend()

    def autolabel(rects, correct_counts, total_counts):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            total = total_counts[idx]
            correct = correct_counts[idx]
            if total == 0 or np.isnan(height):
                label = 'N/A'
                y = 0.02
            else:
                label = f'{correct}/{total}'
                y = height + 0.02
            ax.text(rect.get_x() + rect.get_width()/2., y, label,
                    ha='center', va='bottom', fontsize=9)

    autolabel(rects1, motion_correct, motion_total)
    autolabel(rects2, non_motion_correct, non_motion_total)

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'motion_nonmotion_accuracy_by_duration.png')
    plt.savefig(out_path)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
