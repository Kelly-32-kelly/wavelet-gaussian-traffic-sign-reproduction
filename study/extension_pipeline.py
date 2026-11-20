"""Run the fixed extension sequentially; completed stages are resumable."""
import extension_features
import extension_train
import extension_heads
import extension_native
import sys
import traceback
from pathlib import Path

class Tee:
    def __init__(self,original,stream):self.original=original;self.stream=stream
    def write(self,value):self.original.write(value);self.stream.write(value)
    def flush(self):self.original.flush();self.stream.flush()

def main():
    for dataset,backbone in [('gtsrb','convnext_tiny'),('belgium','vgg16'),('belgium','convnext_tiny')]:
        extension_features.main(dataset,backbone)
        extension_train.main(dataset,backbone)
        extension_heads.tune(dataset,backbone)
        extension_heads.evaluate(dataset,backbone)
    for dataset in ['gtsrb','belgium']:extension_native.main(dataset)
    print('All registered extension evaluations completed',flush=True)

if __name__=='__main__':
    with (Path(__file__).resolve().parent/'extension_run.log').open('a',encoding='utf-8',buffering=1) as log:
        original=sys.stdout;sys.stdout=Tee(original,log)
        try:main()
        except BaseException:
            traceback.print_exc(file=log)
            raise
        finally:sys.stdout=original
