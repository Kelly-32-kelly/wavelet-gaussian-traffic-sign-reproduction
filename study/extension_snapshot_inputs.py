"""Record the immutable experiment inputs and delivery implementation hashes."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    paths=['data/manifest.json','data/belgium/manifest.json','extension_protocol_initial.json',
        'extension_protocol_audit.json','extension_protocol_pre_dedup.json','extension_protocol_pre_solver.json',
        'extension_protocol.json','dino_protocol.json','extension_features.py','extension_train.py',
        'extension_heads.py','extension_native.py','kernel_heads.py','dino_probe.py','references/dinov2_source.json',
        'probe_grid_extension.py','probe_grid_extension_protocol.json']
    records=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in paths]
    folder=ROOT/'extension'/'analysis';folder.mkdir(exist_ok=True)
    (folder/'delivery_input_hashes.json').write_text(json.dumps(dict(
        purpose='Integrity snapshot of retained inputs and delivery implementation after the controlled head runs; not prospective registration',
        files=records),indent=2))
    print('Recorded immutable experiment manifests, protocol versions and delivery code hashes')

if __name__=='__main__':main()
