"""Final integrity checks against the distributor's test label table."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    with (ROOT/'data'/'GT-final_test.csv').open(newline='') as f:
        truth={'test/'+r['Filename']:int(r['ClassId']) for r in csv.DictReader(f,delimiter=';')}
    assert len(truth)==12630
    manifest=json.loads((ROOT/'data'/'manifest.json').read_text())
    assert {r['sample_id']:r['label'] for r in manifest if r['split']=='test'}==truth
    locked=json.loads((ROOT/'results'/'selection'/'locked_configs.json').read_text())
    grid=ROOT/'results'/'selection'/'validation_grid.json'
    assert hashlib.sha256(grid.read_bytes()).hexdigest()==locked['validation_log_sha256']
    checked=[]
    for path in sorted((ROOT/'results').rglob('*predictions.csv')):
        metric=path.with_name(path.name.replace('_predictions.csv','_metrics.json'))
        report=json.loads(metric.read_text())
        assert report['sample_count']==12630 and report['class_count']==43
        assert hashlib.sha256(path.read_bytes()).hexdigest()==report['predictions_sha256']
        correct=0;population={}
        with path.open(newline='') as f:
            for r in csv.DictReader(f):
                identity=r['sample_id'];label=int(r['y_true'])
                assert identity not in population and truth[identity]==label
                population[identity]=label
                correct+=int(int(r['y_pred'])==label)
        assert population==truth
        assert abs(correct/12630-report['accuracy'])<1e-12
        checked.append(dict(predictions=path.relative_to(ROOT).as_posix(),correct=correct,
                            count=12630,sha256=report['predictions_sha256']))
    assert len(checked)==72, f'Expected 63 main and 9 GAP runs, found {len(checked)}'
    result=dict(official_labels_matched=True,run_count=len(checked),test_count=12630,
                lock_unchanged=True,runs=checked)
    (ROOT/'analysis'/'final_integrity_report.json').write_text(json.dumps(result,indent=2))
    print('Verified all 72 prediction populations and accuracies against official test labels')


if __name__=='__main__':
    main()
