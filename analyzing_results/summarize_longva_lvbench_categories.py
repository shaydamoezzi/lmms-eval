import json
import os
import glob
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


QA_JSONL = 'compiled_qa_data/qa.jsonl' if os.path.exists('compiled_qa_data/qa.jsonl') else 'qa.jsonl'
CAT_JSONL = 'compiled_qa_data/motion_qa_categories.jsonl'
CAT_NAMES_JSON = 'compiled_qa_data/category_to_name.json'


def load_qid_to_gid() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(QA_JSONL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Reading qa.jsonl'):
            d = json.loads(line)
            mapping[d['question_id']] = d['global_id']
    return mapping


def load_gid_to_category() -> Dict[str, int]:
    gid2cat: Dict[str, int] = {}
    with open(CAT_JSONL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Reading motion_qa_categories.jsonl'):
            d = json.loads(line)
            gid = d.get('global_id')
            cat = d.get('category')
            if gid is not None and cat is not None:
                gid2cat[gid] = int(cat)
    return gid2cat


def load_category_names() -> Dict[str, str]:
    with open(CAT_NAMES_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data  # keys are strings


def plot_category_distribution(counts_by_cat: Dict[int, Dict[str, int]], cat_names: Dict[str, str], out_png: str, title: str) -> None:
    cat_ids = sorted(counts_by_cat.keys(), key=int)
    totals = [counts_by_cat[c]['total'] for c in cat_ids]
    labels = [cat_names.get(str(c), str(c)) for c in cat_ids]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=totals, color='#4C78A8')
    plt.title(title)
    plt.ylabel('Num Questions')
    plt.xlabel('Motion Category')
    plt.xticks(rotation=20, ha='right')
    for i, v in enumerate(totals):
        plt.text(i, v + max(totals) * 0.01 if totals else 0.02, str(v), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_category_accuracy(counts_by_cat: Dict[int, Dict[str, int]], cat_names: Dict[str, str], out_png: str, title: str) -> None:
    cat_ids = sorted(counts_by_cat.keys(), key=int)
    labels = [cat_names.get(str(c), str(c)) for c in cat_ids]
    totals = [counts_by_cat[c]['total'] for c in cat_ids]
    corrects = [counts_by_cat[c]['correct'] for c in cat_ids]
    accs = [(c / t) if t else 0.0 for c, t in zip(corrects, totals)]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=accs, color='#F58518')
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Motion Category')
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha='right')
    for i, (a, c, t) in enumerate(zip(accs, corrects, totals)):
        label = f"{a:.2%} ({c}/{t})" if t else 'N/A'
        plt.text(i, min(a + 0.03, 0.98), label, ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    qid2gid = load_qid_to_gid()
    gid2cat = load_gid_to_category()
    cat_names = load_category_names()

    result_files = sorted(glob.glob('output/longva-lvbench-*/**/*samples_motion_analysis_bench.jsonl'))

    out_dir = os.path.join('longva-7b', 'lvbench_results')
    os.makedirs(out_dir, exist_ok=True)

    individual_rows: List[List] = []
    cat_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'total': 0})

    for fp in result_files:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f'Processing {os.path.basename(fp)}', leave=False):
                d = json.loads(line)
                bench = d.get('motion_analysis_bench_acc') or {}
                qid = bench.get('question_id')
                pred = bench.get('pred_answer')
                ans = bench.get('answer')
                correct = bench.get('correct')
                if qid is None:
                    continue
                gid = qid2gid.get(qid)
                if gid is None:
                    individual_rows.append([qid, None, None, None, pred, ans, correct])
                    continue
                cat_id = gid2cat.get(gid)
                cat_name = None
                if cat_id is not None:
                    cat_name = cat_names.get(str(cat_id), str(cat_id))
                individual_rows.append([qid, gid, cat_id, cat_name, pred, ans, correct])
                if cat_id is not None and correct is not None:
                    cat_counts[cat_id]['total'] += 1
                    if correct:
                        cat_counts[cat_id]['correct'] += 1

    # write individual CSV
    indiv_csv = os.path.join(out_dir, 'lvbench_individual_responses.csv')
    with open(indiv_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['question_id', 'global_id', 'category_id', 'category_name', 'pred_answer', 'answer', 'correct'])
        w.writerows(individual_rows)

    # write category summary CSV
    cat_csv = os.path.join(out_dir, 'lvbench_category_summary.csv')
    with open(cat_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['category_id', 'category_name', 'correct', 'total', 'accuracy'])
        for cat_id in sorted(cat_counts.keys(), key=int):
            c = cat_counts[cat_id]
            total = c['total']
            correct = c['correct']
            acc = (correct / total) if total else 0.0
            w.writerow([cat_id, cat_names.get(str(cat_id), str(cat_id)), correct, total, f"{acc:.4f}"])

    # plots
    plot_category_distribution(
        cat_counts,
        cat_names,
        os.path.join(out_dir, 'motion_category_distribution_lvbench_longva7b.png'),
        'LongVA-7B LVBench: Motion Category Distribution',
    )
    plot_category_accuracy(
        cat_counts,
        cat_names,
        os.path.join(out_dir, 'accuracy_by_category_lvbench_longva7b.png'),
        'LongVA-7B LVBench: Accuracy by Motion Category',
    )

    print('Saved:', indiv_csv)
    print('Saved:', cat_csv)
    print('Saved plots to', out_dir)


if __name__ == '__main__':
    main()


