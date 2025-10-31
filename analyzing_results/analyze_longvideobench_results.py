import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tqdm import tqdm
import numpy as np


def analyze_results():
    # Resolve data paths (prefer compiled_qa_data if available)
    qa_jsonl = 'compiled_qa_data/qa.jsonl' if os.path.exists('compiled_qa_data/qa.jsonl') else 'qa.jsonl'
    qa_labeled = 'compiled_qa_data/qa_data_labeled.json' if os.path.exists('compiled_qa_data/qa_data_labeled.json') else 'qa_data_labeled.json'

    print("Creating question_id to global_id mapping from qa.jsonl...")
    question_id_to_global_id = {}
    with open(qa_jsonl, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            question_id_to_global_id[data['question_id']] = data['global_id']
    print(f"Found {len(question_id_to_global_id)} mappings in {qa_jsonl}")

    print("Creating global_id to has_motion mapping from qa_data_labeled.json...")
    global_id_to_has_motion = {}
    with open(qa_labeled, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        for item in tqdm(qa_data):
            if 'global_id' in item and 'has_motion' in item:
                global_id_to_has_motion[item['global_id']] = item['has_motion']
    print(f"Found {len(global_id_to_has_motion)} mappings in {qa_labeled}")

    # Auto-discover LongVideoBench result files
    result_files = sorted(glob.glob('output/internvl-longvideobench-*/OpenGVLab__InternVL2-8B/*.jsonl'))
    print(f"Discovered {len(result_files)} LongVideoBench result files")

    print("Processing result files...")
    results = []
    for file_path in result_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {os.path.basename(file_path)}"):
                data = json.loads(line)
                if 'motion_analysis_bench_acc' in data:
                    bench_data = data['motion_analysis_bench_acc']
                    question_id = bench_data.get('question_id')
                    correct = bench_data.get('correct')
                    duration_bucket = bench_data.get('duration_bucket')
                else:
                    question_id = data.get('question_id')
                    correct = data.get('correct')
                    duration_bucket = data.get('duration_bucket')
                if question_id is None or correct is None or duration_bucket is None:
                    continue
                global_id = question_id_to_global_id.get(question_id)
                if global_id:
                    has_motion = global_id_to_has_motion.get(global_id)
                    if has_motion is not None:
                        results.append({'correct': bool(correct), 'has_motion': bool(has_motion), 'duration_bucket': str(duration_bucket).lower()})

    if not results:
        print("No results found. Exiting.")
        return

    df = pd.DataFrame(results)

    print("\n--- LongVideoBench Analysis Report ---")
    overall_accuracy = df['correct'].mean()
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

    motion_df = df[df['has_motion'] == True]
    non_motion_df = df[df['has_motion'] == False]

    if not motion_df.empty:
        motion_accuracy = motion_df['correct'].mean()
        motion_counts = motion_df['correct'].value_counts()
        print(f"Motion Questions Accuracy: {motion_accuracy:.2%}")
        print(f"Motion Correct: {motion_counts.get(True, 0)}, Incorrect: {motion_counts.get(False, 0)}")

    if not non_motion_df.empty:
        non_motion_accuracy = non_motion_df['correct'].mean()
        non_motion_counts = non_motion_df['correct'].value_counts()
        print(f"Non-Motion Questions Accuracy: {non_motion_accuracy:.2%}")
        print(f"Non-Motion Correct: {non_motion_counts.get(True, 0)}, Incorrect: {non_motion_counts.get(False, 0)}")

    print("-----------------------\n")

    print("Generating visualizations...")

    out_dir = os.path.join('internvl-8b', 'longvideobench_results')
    os.makedirs(out_dir, exist_ok=True)

    # Existing plots
    plt.figure(figsize=(8, 6))
    sns.countplot(x='correct', data=df)
    plt.title('LongVideoBench: Overall Distribution of Correct and Incorrect Answers')
    plt.xlabel('Answer Correctness')
    plt.ylabel('Count')
    plt.savefig(os.path.join(out_dir, 'overall_distribution_longvideobench.png'))

    plt.figure(figsize=(10, 6))
    sns.countplot(x='has_motion', hue='correct', data=df)
    plt.title('LongVideoBench: Distribution of Answers by Motion Category')
    plt.xlabel('Has Motion')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Non-Motion', 'Motion'])
    plt.legend(title='Correctness')
    plt.savefig(os.path.join(out_dir, 'motion_distribution_longvideobench.png'))

    accuracy_data = {
        'Category': ['Overall', 'Motion', 'Non-Motion'],
        'Accuracy': [overall_accuracy,
                     motion_df['correct'].mean() if not motion_df.empty else 0,
                     non_motion_df['correct'].mean() if not non_motion_df.empty else 0]
    }
    accuracy_df = pd.DataFrame(accuracy_data)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Category', y='Accuracy', data=accuracy_df)
    plt.title('LongVideoBench: Accuracy by Category')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for index, row in accuracy_df.iterrows():
        plt.text(row.name, row.Accuracy + 0.02, f"{row.Accuracy:.2%}", color='black', ha='center')
    plt.savefig(os.path.join(out_dir, 'accuracy_by_category_longvideobench.png'))

    # New: Duration-bucket grouped chart (short/medium/long x motion/non-motion) with counts
    durations = ['short', 'medium', 'long']
    buckets_present = sorted(set(d for d in df['duration_bucket'].unique() if d in durations), key=lambda x: durations.index(x))
    if not buckets_present:
        print('No valid duration buckets found, skipping duration chart.')
        return

    # Prepare counts per (duration, motion)
    counts = {d: {'motion': {'correct': 0, 'total': 0}, 'non_motion': {'correct': 0, 'total': 0}} for d in buckets_present}
    for _, row in df.iterrows():
        d = row['duration_bucket']
        if d not in counts:
            continue
        motion_key = 'motion' if row['has_motion'] else 'non_motion'
        counts[d][motion_key]['total'] += 1
        if row['correct']:
            counts[d][motion_key]['correct'] += 1

    # Build arrays for plotting
    motion_vals = []
    non_motion_vals = []
    motion_correct = []
    motion_total = []
    non_motion_correct = []
    non_motion_total = []

    for d in buckets_present:
        mc = counts[d]['motion']['correct']; mt = counts[d]['motion']['total']
        nmc = counts[d]['non_motion']['correct']; nmt = counts[d]['non_motion']['total']
        motion_vals.append(mc / mt if mt else np.nan)
        non_motion_vals.append(nmc / nmt if nmt else np.nan)
        motion_correct.append(mc); motion_total.append(mt)
        non_motion_correct.append(nmc); non_motion_total.append(nmt)

    # Plot grouped bars
    x = np.arange(len(buckets_present))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, motion_vals, width, label='Motion', color='#4C78A8')
    rects2 = ax.bar(x + width/2, non_motion_vals, width, label='Non-Motion', color='#F58518')

    ax.set_ylabel('Accuracy')
    ax.set_title('LongVideoBench: Accuracy by Duration Bucket and Motion Type')
    ax.set_xticks(x, [d.capitalize() for d in buckets_present])
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
    out_path = os.path.join(out_dir, 'duration_motion_accuracy_longvideobench.png')
    plt.savefig(out_path)
    print('Saved', out_path)


if __name__ == '__main__':
    analyze_results()
