import json
import os
import csv
from collections import defaultdict
from tqdm import tqdm

QA_JSONL = 'compiled_qa_data/qa.jsonl' if os.path.exists('compiled_qa_data/qa.jsonl') else 'qa.jsonl'
QA_LABELED = 'compiled_qa_data/qa_data_labeled.json' if os.path.exists('compiled_qa_data/qa_data_labeled.json') else 'qa_data_labeled.json'


def main():
    # Load has_motion by global_id
    with open(QA_LABELED, 'r', encoding='utf-8') as f:
        labeled = json.load(f)
    gid_to_motion = {}
    for item in tqdm(labeled, desc='Loading labels'):
        gid = item.get('global_id')
        if gid is None:
            continue
        if 'has_motion' in item:
            gid_to_motion[gid] = bool(item['has_motion'])

    # Aggregate by duration_bucket and motion flag for videomme
    counts = defaultdict(lambda: {'motion': 0, 'non_motion': 0})

    with open(QA_JSONL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Scanning qa.jsonl'):
            d = json.loads(line)
            if d.get('benchmark') != 'videomme':
                continue
            gid = d.get('global_id')
            bucket = (d.get('duration_bucket') or '').lower()
            if not gid or not bucket:
                continue
            hm = gid_to_motion.get(gid)
            if hm is None:
                continue
            key = 'motion' if hm else 'non_motion'
            counts[bucket][key] += 1

    out_dir = os.path.join('internvl-8b', 'videomme_results')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'videomme_duration_motion_distribution.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['duration_bucket', 'motion_count', 'non_motion_count', 'total'])
        for bucket in sorted(counts.keys(), key=lambda x: {'short':0,'medium':1,'long':2}.get(x, 99)):
            motion = counts[bucket]['motion']
            non_motion = counts[bucket]['non_motion']
            total = motion + non_motion
            w.writerow([bucket, motion, non_motion, total])

    print('\nVideoMME: Distribution of has_motion by duration_bucket:')
    for bucket in sorted(counts.keys(), key=lambda x: {'short':0,'medium':1,'long':2}.get(x, 99)):
        motion = counts[bucket]['motion']
        non_motion = counts[bucket]['non_motion']
        total = motion + non_motion
        motion_pct = (motion/total*100) if total else 0.0
        non_motion_pct = (non_motion/total*100) if total else 0.0
        print(f"- {bucket.capitalize()}: motion {motion}/{total} ({motion_pct:.1f}%), non-motion {non_motion}/{total} ({non_motion_pct:.1f}%)")

    print(f"\nSaved CSV: {csv_path}")


if __name__ == '__main__':
    main()




