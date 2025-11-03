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


def compute_motion_nonmotion_acc(pairs: List[Tuple[bool, bool]]) -> Tuple[float, float]:
	motion_pairs = [c for c in pairs if c[1] is True]
	non_motion_pairs = [c for c in pairs if c[1] is False]
	def acc(lst: List[Tuple[bool, bool]]):
		if not lst:
			return np.nan
		num_correct = sum(1 for c, _ in lst if c)
		return num_correct / len(lst)
	return acc(motion_pairs), acc(non_motion_pairs)


def stats_for_model(model: str, qid2gid: Dict[str, str], gid2motion: Dict[str, bool]) -> Dict[str, Tuple[float, float]]:
	"""Return mapping label -> (motion_acc, non_motion_acc) for a model prefix ('longva' or 'internvl')."""
	if model == 'longva':
		nq_files = sorted(glob.glob('output/longva-nextqa-*/**/*samples_motion_analysis_bench.jsonl', recursive=True))
		lvb_files = sorted(glob.glob('output/longva-longvideobench-*/**/*samples_motion_analysis_bench.jsonl', recursive=True))
		lv_files = sorted(glob.glob('output/longva-lvbench-*/**/*samples_motion_analysis_bench.jsonl', recursive=True))
		ego_files = sorted(glob.glob('output/longva-egoschema-*/**/*samples_motion_analysis_bench.jsonl', recursive=True))
	else:
		nq_files = sorted(glob.glob('output/internvl-nextqa-*/**/*samples_motion_analysis_bench.jsonl', recursive=True))
		lvb_files = sorted(glob.glob('output/internvl-longvideobench-*/**/*samples_motion_analysis_bench.jsonl', recursive=True))
		lv_files = sorted(glob.glob('output/internvl-lvbench-*/**/*samples_motion_analysis_bench.jsonl', recursive=True))
		ego_files = sorted(glob.glob('output/internvl-egoschema-*/**/*samples_motion_analysis_bench.jsonl', recursive=True))

	labels = [
		('Short - NextQA', [('pairs', collect_results(nq_files, qid2gid, gid2motion), 'short')]),
		('Short - LVB-S', [('triplets', collect_results_with_duration(lvb_files, qid2gid, gid2motion), 'short')]),
		('Medium - EgoSchema', [('pairs', collect_results(ego_files, qid2gid, gid2motion), 'medium')]),
		('Medium - LVB-M', [('triplets', collect_results_with_duration(lvb_files, qid2gid, gid2motion), 'medium')]),
		('Long - LVB-L', [('triplets', collect_results_with_duration(lvb_files, qid2gid, gid2motion), 'long')]),
		('Long - LVBench', [('pairs', collect_results(lv_files, qid2gid, gid2motion), 'long')]),
	]

	accs: Dict[str, Tuple[float, float]] = {}
	for name, sources in labels:
		pairs: List[Tuple[bool, bool]] = []
		for kind, data, bucket in sources:
			if kind == 'pairs':
				pairs.extend(data)
			else:
				pairs.extend([(c, m) for (c, m, d) in data if d == bucket])
		if not pairs:
			continue
		accs[name] = compute_motion_nonmotion_acc(pairs)
	return accs


def main() -> None:
	qa_jsonl = 'compiled_qa_data/qa.jsonl' if os.path.exists('compiled_qa_data/qa.jsonl') else 'qa.jsonl'
	qa_labeled = 'compiled_qa_data/qa_data_labeled.json' if os.path.exists('compiled_qa_data/qa_data_labeled.json') else 'qa_data_labeled.json'

	qid2gid = load_question_to_global_id(qa_jsonl)
	gid2motion = load_global_id_to_motion(qa_labeled)

	order = [
		'Short - NextQA',
		'Short - LVB-S',
		'Medium - EgoSchema',
		'Medium - LVB-M',
		'Long - LVB-L',
		'Long - LVBench',
	]

	longva_stats = stats_for_model('longva', qid2gid, gid2motion)
	internvl_stats = stats_for_model('internvl', qid2gid, gid2motion)

	labels = [lbl for lbl in order if lbl in longva_stats or lbl in internvl_stats]
	x = np.arange(len(labels))

	# Build arrays (fill with np.nan when missing)
	lv_motion = np.array([longva_stats.get(lbl, (np.nan, np.nan))[0] for lbl in labels], dtype=float)
	lv_non_motion = np.array([longva_stats.get(lbl, (np.nan, np.nan))[1] for lbl in labels], dtype=float)
	iv_motion = np.array([internvl_stats.get(lbl, (np.nan, np.nan))[0] for lbl in labels], dtype=float)
	iv_non_motion = np.array([internvl_stats.get(lbl, (np.nan, np.nan))[1] for lbl in labels], dtype=float)

	plt.figure(figsize=(14, 6))
	# Motion
	mask = np.isfinite(lv_motion)
	plt.plot(x[mask], lv_motion[mask], color='#4C78A8', marker='o', linewidth=2.5, linestyle='-', label='Motion (LongVA-7B)')
	mask = np.isfinite(iv_motion)
	plt.plot(x[mask], iv_motion[mask], color='#4C78A8', marker='^', linewidth=2.5, linestyle='--', label='Motion (InternVL-8B)')
	# Non-motion
	mask = np.isfinite(lv_non_motion)
	plt.plot(x[mask], lv_non_motion[mask], color='#F58518', marker='s', linewidth=2.5, linestyle='-', label='Non-Motion (LongVA-7B)')
	mask = np.isfinite(iv_non_motion)
	plt.plot(x[mask], iv_non_motion[mask], color='#F58518', marker='v', linewidth=2.5, linestyle='--', label='Non-Motion (InternVL-8B)')

	plt.xticks(x, labels, rotation=25, ha='right')
	plt.ylim(0, 1)
	plt.ylabel('Accuracy')
	plt.title('Motion vs Non-Motion Accuracy by Duration: LongVA-7B vs InternVL-8B')
	plt.grid(alpha=0.25, linestyle='--')
	plt.legend()
	plt.tight_layout()

	# Save to both model comparison folders and a root folder
	paths = [
		os.path.join('comparison_results', 'motion_nonmotion_accuracy_by_duration_model_comparison.png'),
		os.path.join('longva-7b', 'comparison_results', 'motion_nonmotion_accuracy_by_duration_model_comparison.png'),
		os.path.join('internvl-8b', 'comparison_results', 'motion_nonmotion_accuracy_by_duration_model_comparison.png'),
	]
	for p in paths:
		os.makedirs(os.path.dirname(p), exist_ok=True)
		plt.savefig(p, dpi=300)
		print('Saved', p)


if __name__ == '__main__':
	main()
