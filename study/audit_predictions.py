"""Audit saved single-label predictions without training a model.

CSV columns: sample_id,y_true,y_pred,score_0,...,score_{C-1}.
Optional manifest columns: sample_id,split,group_id,sha256.
Scores must be the actual exported class scores, not fabricated probabilities.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def average_precision(labels, scores):
    positives = sum(labels)
    if not positives:
        raise ValueError('AP undefined for a class with no positive test samples')
    ranked = sorted(zip(scores, labels), key=lambda row: row[0], reverse=True)
    tp = seen = 0
    previous_recall = ap = 0.0
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        end = index
        while end < len(ranked) and ranked[end][0] == threshold:
            tp += ranked[end][1]
            seen += 1
            end += 1
        recall = tp / positives
        ap += (recall - previous_recall) * tp / seen
        previous_recall = recall
        index = end
    return ap


def audit_rows(rows, classes, allow_missing_test_classes=False):
    if classes < 2 or not rows:
        raise ValueError('At least two classes and one prediction row are required')
    ids = [row['sample_id'] for row in rows]
    if any(not identity for identity in ids) or len(set(ids)) != len(ids):
        raise ValueError('Test sample identifiers must be nonempty and unique')
    truth, prediction, scores = [], [], []
    confusion = [[0] * classes for _ in range(classes)]
    for row in rows:
        y, p = int(row['y_true']), int(row['y_pred'])
        if not (0 <= y < classes and 0 <= p < classes):
            raise ValueError('Labels must be in range 0..classes-1')
        values = [float(row[f'score_{c}']) for c in range(classes)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Class scores must be finite')
        score_prediction = max(range(classes), key=lambda c: values[c])
        if score_prediction != p:
            raise ValueError('Prediction is inconsistent with scores and the fixed tie rule')
        truth.append(y)
        prediction.append(p)
        scores.append(values)
        confusion[y][p] += 1
    support = [sum(row) for row in confusion]
    if any(count == 0 for count in support) and not allow_missing_test_classes:
        raise ValueError('Every task class must have positive test support')
    per_class = []
    for c in range(classes):
        tp = confusion[c][c]
        predicted = sum(row[c] for row in confusion)
        fp, fn = predicted - tp, support[c] - tp
        precision = tp / predicted if predicted else 0.0
        recall = tp / support[c] if support[c] else None
        f1 = 2 * tp / (2 * tp + fp + fn) if support[c] else None
        ap = average_precision([int(y == c) for y in truth], [s[c] for s in scores]) if support[c] else None
        per_class.append(dict(class_id=c, support=support[c], precision=precision,
                              recall=recall, f1=f1, average_precision=ap))
    accuracy = sum(y == p for y, p in zip(truth, prediction)) / len(rows)
    supported=[row for row in per_class if row['support']>0]
    macro = {key: sum(row[key] for row in supported) / len(supported)
             for key in ('precision', 'recall', 'f1', 'average_precision')}
    balanced = len(set(support)) == 1
    if balanced and not math.isclose(accuracy, macro['recall'], abs_tol=1e-12):
        raise AssertionError('Equal-support recall/accuracy identity failed')
    return dict(sample_count=len(rows), class_count=classes, accuracy=accuracy,
                macro=macro, equal_support=balanced, per_class=per_class,
                macro_class_ids=[row['class_id'] for row in supported],
                macro_class_count=len(supported),
                absent_test_class_ids=[c for c,n in enumerate(support) if not n],
                confusion_matrix=confusion,
                ap_estimator='non-interpolated, grouped score ties')


def audit_manifest(rows, predicted_ids, allow_within_split_duplicates=False):
    ids = [row['sample_id'] for row in rows]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError('Manifest sample IDs must be nonempty and globally unique')
    valid_splits = {'train', 'validation', 'test'}
    if any(row['split'] not in valid_splits for row in rows):
        raise ValueError('Manifest split must be train, validation, or test')
    test_ids = {row['sample_id'] for row in rows if row['split'] == 'test'}
    if test_ids != set(predicted_ids):
        raise ValueError('Prediction IDs must exactly match the manifest test population')
    checks = {}
    for key in ('group_id', 'sha256'):
        membership = defaultdict(set)
        occurrences = Counter()
        present = 0
        for row in rows:
            value = row.get(key, '').strip()
            if value:
                present += 1
                membership[value].add(row['split'])
                occurrences[value] += 1
        cross_split = [value for value, splits in membership.items() if len(splits) > 1]
        if cross_split:
            raise ValueError(f'Cross-split overlap detected in {key}: {len(cross_split)} keys')
        checks[key] = dict(rows_with_value=present, total_rows=len(rows),
                           coverage=present / len(rows) if rows else 0,
                           complete=present == len(rows), cross_split_overlap=0,
                           repeated_keys=sum(n > 1 for n in occurrences.values()))
    if checks['sha256']['repeated_keys'] and not allow_within_split_duplicates:
        raise ValueError('Duplicate image-content hashes found; resolve before evaluation')
    return dict(split_counts=dict(Counter(row['split'] for row in rows)), checks=checks,
                within_split_duplicates_permitted=allow_within_split_duplicates,
                note='Absent hashes/groups are unverified, not evidence of no leakage')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions')
    parser.add_argument('--classes', required=True, type=int)
    parser.add_argument('--manifest')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = read_csv(args.predictions)
    report = audit_rows(rows, args.classes)
    report['manifest_audit'] = (audit_manifest(read_csv(args.manifest),
        [row['sample_id'] for row in rows]) if args.manifest else 'Not supplied; split unverified')
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Audited {report['sample_count']} predictions; report saved")


if __name__ == '__main__':
    main()
