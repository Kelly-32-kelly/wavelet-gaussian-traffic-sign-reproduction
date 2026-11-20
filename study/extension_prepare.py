"""Create a filename-prefix-separated validation split of official BelgiumTSC."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'belgium'

def main():
    rows=[];invalid_roi=0;dimension_mismatch=0;seen={};duplicate_annotations=[];annotation_counts=Counter()
    for partition,split in [('Training','train'),('Testing','test')]:
        for table in sorted((DATA/partition).glob('*/GT-*.csv')):
            with table.open(newline='') as stream:
                for r in csv.DictReader(stream,delimiter=';'):
                    path=table.parent/r['Filename']; label=int(r['ClassId'])
                    annotation_counts[split]+=1
                    identity=f'belgium/{partition}/'+path.relative_to(DATA/partition).as_posix()
                    if identity in seen:
                        assert seen[identity]==r,'Conflicting annotation rows for the same file'
                        duplicate_annotations.append(dict(sample_id=identity,annotation=r,partition=partition))
                        continue
                    seen[identity]=r
                    with Image.open(path) as im:
                        pixels=np.asarray(im.convert('RGB'))
                        pixel_sha=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
                        width,height=im.size
                    roi=[int(r[k]) for k in ['Roi.X1','Roi.Y1','Roi.X2','Roi.Y2']]
                    invalid_roi+=not(0<=roi[0]<=roi[2]<width and 0<=roi[1]<=roi[3]<height)
                    dimension_mismatch+=(int(r['Width']),int(r['Height']))!=(width,height)
                    rows.append(dict(sample_id=identity,
                        path=path.relative_to(ROOT).as_posix(),label=label,
                        group_id=f'{label}:{r["Filename"].split("_")[0]}',
                        roi=roi,model_roi=[0,0,width-1,height-1],image_width=width,image_height=height,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pixel_sha256=pixel_sha,split=split))
    test_hashes={r['pixel_sha256'] for r in rows if r['split']=='test'}
    test_groups={r['group_id'] for r in rows if r['split']=='test'}
    excluded_groups={r['group_id'] for r in rows if r['split']=='train' and
                     (r['pixel_sha256'] in test_hashes or r['group_id'] in test_groups)}
    excluded=[r for r in rows if r['split']=='train' and r['group_id'] in excluded_groups]
    original=Counter(r['split'] for r in rows)
    for partition,split in [('Training','train'),('Testing','test')]:
        actual={path.relative_to(ROOT).as_posix() for path in (DATA/partition).rglob('*.ppm')}
        assert actual=={r['path'] for r in rows if r['split']==split}
    rows=[r for r in rows if r not in excluded]
    parents={r['group_id']:r['group_id'] for r in rows if r['split']=='train'}
    def find(g):
        while parents[g]!=g:
            parents[g]=parents[parents[g]];g=parents[g]
        return g
    first={}
    for r in rows:
        if r['split']!='train': continue
        h=r['pixel_sha256']
        if h in first:
            other=first[h]
            assert other['label']==r['label'],'Conflicting duplicate labels'
            a,b=find(r['group_id']),find(other['group_id']);parents[max(a,b)]=min(a,b)
        else: first[h]=r
    byclass=defaultdict(set)
    for r in rows:
        if r['split']=='train': byclass[r['label']].add(find(r['group_id']))
    rng=np.random.default_rng(20261014);valgroups=set()
    for c,groups in sorted(byclass.items()):
        order=sorted(groups);rng.shuffle(order)
        count=min(len(order)-1,max(1,round(.2*len(order))))
        valgroups.update(order[:count])
    for r in rows:
        if r['split']=='train' and find(r['group_id']) in valgroups:r['split']='validation'
    hash_splits=defaultdict(set)
    for r in rows:hash_splits[r['pixel_sha256']].add(r['split'])
    assert all(len(s)==1 for s in hash_splits.values())
    groups={s:{r['group_id'] for r in rows if r['split']==s} for s in ['train','validation','test']}
    assert not groups['train']&groups['validation'] and not groups['train']&groups['test'] and not groups['validation']&groups['test']
    assert set(byclass)==set(range(62))
    assert len({r['sample_id'] for r in rows})==len(rows)
    counts={s:dict(sorted(Counter(r['label'] for r in rows if r['split']==s).items())) for s in ['train','validation','test']}
    summary=dict(original_population=dict(original),population={s:sum(v.values()) for s,v in counts.items()},
        original_annotation_row_counts=dict(annotation_counts),duplicate_annotation_rows=len(duplicate_annotations),
        counts=counts,class_count=62,split_seed=20261014,excluded_training_groups=sorted(excluded_groups),
        excluded_training_images=len(excluded),cross_split_identical_pixels=0,
        exclusion_reason='Conservative exclusion of training filename-prefix groups also present in official test, or identical decoded pixels',
        cross_split_filename_prefix_groups=0,
        original_annotation_dimension_mismatches=dimension_mismatch,original_annotation_out_of_image_rois=invalid_roi,
        model_crop='entire supplied classification crop; original inconsistent ROI metadata is retained but unused',
        group_rule='ClassId plus filename prefix before underscore; physical independence is not independently verified',
        classes_without_validation_examples=sorted(set(range(62))-set(counts['validation'])))
    (DATA/'manifest.json').write_text(json.dumps(rows,indent=2))
    (DATA/'split_summary.json').write_text(json.dumps(summary,indent=2))
    (DATA/'excluded_training_groups.json').write_text(json.dumps(excluded,indent=2))
    (DATA/'duplicate_annotation_rows.json').write_text(json.dumps(duplicate_annotations,indent=2))
    with (DATA/'split_manifest.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['sample_id','split','group_id','sha256','pixel_sha256']);w.writeheader()
        w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
