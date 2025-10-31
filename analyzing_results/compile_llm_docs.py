import json
import os
import glob
from typing import Dict, List, Tuple
from tqdm import tqdm


QA_JSONL = 'compiled_qa_data/qa.jsonl' if os.path.exists('compiled_qa_data/qa.jsonl') else 'qa.jsonl'
QA_LABELED = 'compiled_qa_data/qa_data_labeled.json' if os.path.exists('compiled_qa_data/qa_data_labeled.json') else 'qa_data_labeled.json'


def load_question_metadata() -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """
    Returns:
      - question_id_to_global_id
      - question_id_to_meta: {'question': str, 'options': dict, 'benchmark': str}
    """
    qid2gid: Dict[str, str] = {}
    qid2meta: Dict[str, Dict] = {}
    with open(QA_JSONL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Reading qa.jsonl'):
            d = json.loads(line)
            qid = d.get('question_id')
            if not qid:
                continue
            qid2gid[qid] = d.get('global_id')
            qid2meta[qid] = {
                'question': d.get('question'),
                'options': d.get('options') or {},
                'benchmark': d.get('benchmark'),
                'duration_bucket': (d.get('duration_bucket') or ''),
            }
    return qid2gid, qid2meta


def load_global_motion() -> Dict[str, bool]:
    gid2motion: Dict[str, bool] = {}
    if not os.path.exists(QA_LABELED):
        return gid2motion
    with open(QA_LABELED, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in tqdm(data, desc='Reading qa_data_labeled.json'):
            gid = item.get('global_id')
            if gid is None:
                continue
            if 'has_motion' in item:
                gid2motion[gid] = bool(item['has_motion'])
    return gid2motion


def collect_model_results(model_prefix: str) -> Dict[str, bool]:
    """Collect latest correctness per question_id across all benchmarks for a model.
    model_prefix examples: 'longva' or 'internvl'
    """
    patterns = [
        f'output/{model_prefix}-lvbench-*/**/*samples_motion_analysis_bench.jsonl',
        f'output/{model_prefix}-longvideobench-*/**/*samples_motion_analysis_bench.jsonl',
        f'output/{model_prefix}-nextqa-*/**/*samples_motion_analysis_bench.jsonl',
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(files)

    qid2correct: Dict[str, bool] = {}
    for fp in files:
        if not os.path.exists(fp):
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                bench = d.get('motion_analysis_bench_acc') or {}
                qid = bench.get('question_id')
                corr = bench.get('correct')
                if qid is None or corr is None:
                    continue
                # latest wins due to sorted order
                qid2correct[qid] = bool(corr)
    return qid2correct


def render_markdown(entries: List[Tuple[str, Dict]], title: str) -> str:
    """entries: list of (question_id, payload) with payload containing fields used below"""
    lines: List[str] = [f"### {title}", ""]
    for qid, p in entries:
        q = p.get('question') or ''
        opts = p.get('options') or {}
        has_motion = p.get('has_motion')
        correct = p.get('correct')
        duration_bucket = (p.get('duration_bucket') or '').lower() or 'unknown'
        lines.append(f"- **QID**: {qid} | **motion**: {'Y' if has_motion else 'N'} | **correct**: {correct} | **duration**: {duration_bucket}")
        lines.append("")
        lines.append(f"  - **Question**: {q}")
        if isinstance(opts, dict):
            # keep option order A..J if present
            for key in ["A","B","C","D","E","F","G","H","I","J"]:
                if key in opts and opts[key] is not None:
                    lines.append(f"  - {key}. {opts[key]}")
        lines.append("")
    return "\n".join(lines)


def write_docs_for_model(model_prefix: str, out_base_dir: str) -> None:
    qid2gid, qid2meta = load_question_metadata()
    gid2motion = load_global_motion()
    qid2correct = collect_model_results(model_prefix)

    # Build payloads
    payload: Dict[str, Dict] = {}
    for qid, correct in qid2correct.items():
        meta = qid2meta.get(qid)
        if not meta:
            continue
        gid = qid2gid.get(qid)
        has_motion = gid2motion.get(gid)
        payload[qid] = {
            'question': meta.get('question'),
            'options': meta.get('options'),
            'benchmark': meta.get('benchmark'),
            'has_motion': bool(has_motion) if has_motion is not None else False,
            'correct': bool(correct),
            'duration_bucket': meta.get('duration_bucket'),
        }

    # Partition
    motion_entries = [(qid, p) for qid, p in payload.items() if p['has_motion'] is True]
    non_motion_entries = [(qid, p) for qid, p in payload.items() if p['has_motion'] is False]
    correct_entries = [(qid, p) for qid, p in payload.items() if p['correct'] is True]
    incorrect_entries = [(qid, p) for qid, p in payload.items() if p['correct'] is False]

    os.makedirs(out_base_dir, exist_ok=True)
    with open(os.path.join(out_base_dir, 'motion_questions.md'), 'w', encoding='utf-8') as f:
        f.write(render_markdown(sorted(motion_entries), f"{model_prefix.upper()}: Motion Questions"))
    with open(os.path.join(out_base_dir, 'non_motion_questions.md'), 'w', encoding='utf-8') as f:
        f.write(render_markdown(sorted(non_motion_entries), f"{model_prefix.upper()}: Non-Motion Questions"))
    with open(os.path.join(out_base_dir, 'correct_questions.md'), 'w', encoding='utf-8') as f:
        f.write(render_markdown(sorted(correct_entries), f"{model_prefix.upper()}: Correct Questions"))
    with open(os.path.join(out_base_dir, 'incorrect_questions.md'), 'w', encoding='utf-8') as f:
        f.write(render_markdown(sorted(incorrect_entries), f"{model_prefix.upper()}: Incorrect Questions"))

    print('Wrote docs to', out_base_dir)


def main() -> None:
    write_docs_for_model('longva', os.path.join('analysis_docs', 'longva-7b'))
    write_docs_for_model('internvl', os.path.join('analysis_docs', 'internvl-8b'))


if __name__ == '__main__':
    main()


