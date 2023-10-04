import argparse
from contextlib import contextmanager
from functools import partial
import gc
import logging
from timeit import default_timer
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import islice, product
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda import OutOfMemoryError
from perceiver_pytorch import Perceiver as LucidrainsPerceiver
from flash_perceiver import Perceiver


sns.set_theme()

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()


def build_lucidrains_perceiver(config, **kwargs):
    return LucidrainsPerceiver(
        input_channels=config['input_dim'],
        input_axis=1,
        num_freq_bands=None,
        max_freq=None,
        num_latents=config['num_latents'],
        latent_dim=config['latent_dim'],
        depth=config['depth'],
        final_classifier_head=False,
        fourier_encode_data=False,
        **kwargs
    )


def build_flash_perceiver(config, **kwargs):
    return Perceiver(
        input_dim=config['input_dim'],
        depth=config['depth'],
        num_latents=config['num_latents'],
        latent_dim=config['latent_dim'],
        **kwargs
    )


num_batches = 100

default_config = {
    'batch_size': 256,
    'input_dim': 128,
    'input_size': 512,
    'depth': 8,
    'latent_dim': 256,
    'num_latents': 256
}
    
benchmark_configs = [
    # Use different batch size to prevent OOM errors
    {
        'input_size': [128, 256, 512],
        'batch_size': 256,
    },
    {
        'input_size': [1024, 2048, 4096],
        'batch_size': 128,
    },
    {
        'input_size': [8192],
        'batch_size': 64,
    },
    {
        'input_size': [16384],
        'batch_size': 48,
    },
    # {
    #     'input_size': [2048, 4096, 8196, 16392],
    #     'batch_size': 128,
    # },

    # {'depth': [6, 12, 24]},
    # {'num_latents': [32, 64, 128, 256, 512]},
    # {'masking_rate': [0.0, 0.2, 0.4, 0.6, 0.8]},
]

models = {
    'perceiver-pytorch': build_lucidrains_perceiver,
    'flash-perceiver': build_flash_perceiver,
}


class DummyDataset(IterableDataset):
    def __init__(self, dim, seq_len, batch_size, mask_rate=None):
        self.dim = dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.mask_rate = mask_rate

    def __iter__(self):
        while True:
            yield torch.randn(self.batch_size, self.seq_len, self.dim)


def default_list(o):
    if isinstance(o, list):
        return o
    elif isinstance(o, tuple):
        return list(o)
    else:
        return [o]


def create_configs(configs):
    for config in configs:
        config = {k: default_list(v) for k, v in config.items()}
        config_tuples = [[(k, v) for v in vs] for k, vs in config.items()]
        yield from map(dict, product(*config_tuples))


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start

    yield lambda: elapser()

    end = default_timer()
    elapser = lambda: end - start


def benchmark_single(model_factory, config, pbar=True, handle_oom=False):
    orig_config = config

    while True:
        try:
            config = {**default_config, **orig_config}

            model = model_factory(config)
            dataset = DummyDataset(config['input_dim'], config['input_size'], config['batch_size'])
                                            
            data_loader = DataLoader(dataset, batch_size=None)
            batches = list(islice(data_loader, num_batches))

            def run_epoch(_batches):
                for batch in _batches:
                    out = model(batch)
                    out.mean().backward()
                
                torch.cuda.synchronize()
            
            with torch.autocast('cuda'):
                # Do some warmup first
                run_epoch(batches[:1])

                if pbar:
                    batches = tqdm(batches[1:])

                with elapsed_timer() as elapser:
                    run_epoch(batches)
                    return elapser()
        
        except OutOfMemoryError:
            if not handle_oom:
                raise

            logger.info('OOM, retrying; reducing batch size from '
                       f'{config["batch_size"]} to {config["batch_size"] // 2}')
            orig_config["batch_size"] //= 2


def reset_all():
    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    

def main(args):
    torch.set_default_device(args.device)
    torch.set_default_dtype(torch.float16)

    if args.quiet:
        logger.removeHandler(logger.handlers[0])

    results = []

    for model_name, model_factory in models.items():
        logger.info(f'Benchmarking {model_name}')

        for config in create_configs(benchmark_configs):
            logger.info(config)

            reset_all()

            run_time = benchmark_single(model_factory, config, pbar=not args.quiet)

            mem = torch.cuda.max_memory_allocated() / ((2 ** 20) * 1000)

            samples_sum = config['batch_size'] * num_batches

            results.append({
                **config,
                'model': model_name,
                'run_time': run_time,
                'peak_memory': mem,
                'it_per_sec': samples_sum / run_time,
                'time_per_it': run_time / samples_sum,
            })

            reset_all()
    
    df = pd.DataFrame(results)
    df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='benchmark_results.csv')
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--device', default='cuda')
    
    main(parser.parse_args())
