import hashlib
from pathlib import Path
import scanpy as sc

CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)

def _cache_path(prefix: str, *args, ext: str = 'h5ad') -> Path:
    """
    Generate a unique cache filename under `CACHE_DIR` based on `prefix`
    and a hash of the stringified args.
    """
    key = hashlib.md5('_'.join(map(str, args)).encode()).hexdigest()
    return CACHE_DIR / f'{prefix}_{key}.{ext}'

def _load_or_run(cache_file: Path, compute_fn):
    """
    If `cache_file` exists, load it with Scanpy; otherwise run `compute_fn`,
    save to cache and return the result.
    """
    if cache_file.exists():
        return sc.read_h5ad(cache_file)
    adata = compute_fn()
    adata.write_h5ad(cache_file)
    return adata
