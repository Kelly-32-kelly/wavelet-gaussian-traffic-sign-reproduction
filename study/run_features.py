import hashlib
import json
import os
import time
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache'
TRANSFORM = Compose([Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                     ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row['path']) as image:
            image = image.convert('RGB')
            x1, y1, x2, y2 = row['roi']
            if not (0 <= x1 <= x2 < image.width and 0 <= y1 <= y2 < image.height):
                raise ValueError('ROI outside supplied image: ' + row['sample_id'])
            image = image.crop((x1, y1, x2 + 1, y2 + 1))
            return TRANSFORM(image)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required by the registered experiment')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    CACHE.mkdir(exist_ok=True)
    torch.hub.set_dir(str(ROOT / 'torch_cache'))
    source = ROOT / 'data' / 'manifest.json'
    rows = json.loads(source.read_text())
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    progress_path = CACHE / 'features_progress.json'
    output_path = CACHE / 'vgg_conv.npy'
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['manifest_sha256'] != fingerprint:
            raise ValueError('Manifest changed; incompatible feature cache')
        offset = progress['processed']
        result = np.load(output_path, mmap_mode='r+')
    else:
        offset = 0
        result = np.lib.format.open_memmap(output_path, mode='w+', dtype='float16',
                                         shape=(len(rows), 25088))
    if offset == len(rows):
        print('Existing feature cache complete', flush=True)
        return
    print('Loading ImageNet VGG16 convolutional weights', flush=True)
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval().cuda()
    torch.save(model.state_dict(), CACHE/'frozen_conv.pt')
    loader = DataLoader(Images(rows[offset:]), batch_size=24, shuffle=False,
                        num_workers=2, pin_memory=True)
    start = time.perf_counter()
    last = start
    done = offset
    with torch.inference_mode():
        for batch in loader:
            batch = batch.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                values = model(batch).flatten(1)
            count = len(batch)
            result[done:done + count] = values.cpu().numpy()
            done += count
            now = time.perf_counter()
            if now - last > 25 or done == len(rows):
                result.flush()
                progress_path.write_text(json.dumps(dict(processed=done,
                    manifest_sha256=fingerprint, device=torch.cuda.get_device_name(0),
                    torch=torch.__version__, weights='VGG16_IMAGENET1K_V1',
                    conv_compute='CUDA autocast float16', cache_dtype='float16',
                    seconds_this_segment=now - start, completed=done == len(rows)), indent=2))
                print(f'Features {done}/{len(rows)}; segment {now-start:.1f}s', flush=True)
                last = now
    result.flush()
    print('All frozen convolutional features saved', flush=True)


if __name__ == '__main__':
    main()
