import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm


def analyze_results():
    print("Creating question_id to global_id mapping from qa.jsonl...")
    question_id_to_global_id = {}
    with open('qa.jsonl', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            question_id_to_global_id[data['question_id']] = data['global_id']
    print(f"Found {len(question_id_to_global_id)} mappings in qa.jsonl")

    print("Creating global_id to has_motion mapping from qa_data_labeled.json...")
    global_id_to_has_motion = {}
    with open('qa_data_labeled.json', 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        for item in tqdm(qa_data):
            if 'global_id' in item and 'has_motion' in item:
                global_id_to_has_motion[item['global_id']] = item['has_motion']
    print(f"Found {len(global_id_to_has_motion)} mappings in qa_data_labeled.json")

    result_files = [
        'output/internvl-lvbench-1/OpenGVLab__InternVL2-8B/20251028_232855_samples_motion_analysis_bench.jsonl',
        'output/internvl-lvbench-2/OpenGVLab__InternVL2-8B/20251028_233849_samples_motion_analysis_bench.jsonl',
        'output/internvl-lvbench-3/OpenGVLab__InternVL2-8B/20251029_005716_samples_motion_analysis_bench.jsonl',
        'output/internvl-lvbench-4/OpenGVLab__InternVL2-8B/20251029_010159_samples_motion_analysis_bench.jsonl'
    ]

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
                else:
                    question_id = data.get('question_id')
                    correct = data.get('correct')
                if question_id is None or correct is None:
                    continue
                global_id = question_id_to_global_id.get(question_id)
                if global_id:
                    has_motion = global_id_to_has_motion.get(global_id)
                    if has_motion is not None:
                        results.append({'correct': bool(correct), 'has_motion': bool(has_motion)})

    if not results:
        print("No results found. Exiting.")
        return

    df = pd.DataFrame(results)

    print("\n--- Analysis Report ---")
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

    plt.figure(figsize=(8, 6))
    sns.countplot(x='correct', data=df)
    plt.title('Overall Distribution of Correct and Incorrect Answers')
    plt.xlabel('Answer Correctness')
    plt.ylabel('Count')
    plt.savefig('overall_distribution.png')

    plt.figure(figsize=(10, 6))
    sns.countplot(x='has_motion', hue='correct', data=df)
    plt.title('Distribution of Answers by Motion Category')
    plt.xlabel('Has Motion')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Non-Motion', 'Motion'])
    plt.legend(title='Correctness')
    plt.savefig('motion_distribution.png')

    accuracy_data = {
        'Category': ['Overall', 'Motion', 'Non-Motion'],
        'Accuracy': [overall_accuracy,
                     motion_df['correct'].mean() if not motion_df.empty else 0,
                     non_motion_df['correct'].mean() if not non_motion_df.empty else 0]
    }
    accuracy_df = pd.DataFrame(accuracy_data)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Category', y='Accuracy', data=accuracy_df)
    plt.title('Accuracy by Category')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for index, row in accuracy_df.iterrows():
        plt.text(row.name, row.Accuracy + 0.02, f"{row.Accuracy:.2%}", color='black', ha='center')
    plt.savefig('accuracy_by_category.png')

    print("Saved overall_distribution.png, motion_distribution.png, accuracy_by_category.png")


if __name__ == '__main__':
    analyze_results()
