# Adapted for GPU-Tensor input https://github.com/noosphaerenbauer/tiny-knn/blob/main/tiny_knn/api.py
import os
import time
from typing import Tuple, Optional
import numpy as np  # type: ignore
import torch
import torch.nn.functional as F

def _get_free_cuda_mem() -> Tuple[int, int]:
    """Returns the free and total CUDA memory in bytes."""
    try:
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)
    except Exception:
        return 0, 0


def _estimate_batch_size(Q: int, dim: int, emb_bytes: int, safety_frac: float = 0.5, max_batch_size: int = 131072) -> int:
    """
    Estimate a safe batch size to fit in GPU memory.
    This is a heuristic that allocates a fraction of memory for the query batch.
    """
    free_mem, _ = _get_free_cuda_mem()
    if free_mem <= 0:
        # CPU or unknown: pick a conservative default
        return min(Q, max_batch_size)

    usable = int(free_mem * safety_frac)

    # Allocate ~20% of usable memory to the query batch tensor
    query_mem_limit = int(usable * 0.2)

    # Calculate batch size based on memory for a single query vector
    bs = query_mem_limit // (dim * emb_bytes + 1)

    # Clamp the batch size to reasonable limits
    return max(1, min(bs, Q, max_batch_size))


def _estimate_doc_chunk_rows(D: int, dim: int, batch_size: int, emb_bytes: int, score_bytes: int, safety_frac: float = 0.5, max_chunk_size: int = 131072) -> int:
    """
    Estimate a safe number of document rows to load on GPU per chunk so that:
      - result matrix (batch_size x rows) fits comfortably
      - doc chunk tensor (rows x dim) fits comfortably
    Uses a fraction of currently free memory for safety.
    """
    free_mem, _ = _get_free_cuda_mem()
    if free_mem <= 0:
        # CPU or unknown: pick a conservative default
        return max(1, min(D, max_chunk_size))

    usable = int(free_mem * safety_frac)
    if usable <= 0:
        return max(1, min(D, max_chunk_size))

    # Memory for one row of docs and one row of scores
    mem_per_row_docs = dim * emb_bytes
    mem_per_row_scores = batch_size * score_bytes

    # Total memory per row is the sum of these
    total_mem_per_row = mem_per_row_docs + mem_per_row_scores

    # Calculate how many rows can fit
    rows = usable // (total_mem_per_row + 1)

    # Clip to dataset size and a reasonable maximum chunk size
    return max(1, min(D, max_chunk_size, rows))


def _to_device_chunk(t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Moves a CPU torch tensor chunk to the target device with proper dtype, using pinned memory on CUDA."""
    if t.device == device:
        return t if t.dtype == dtype else t.to(dtype=dtype)

    if device.type == "cuda" and t.device.type == "cpu":
        t = t.pin_memory()
        return t.to(device=device, dtype=dtype, non_blocking=True)

    return t.to(device=device, dtype=dtype)



def load_array(path: str):
    # Load tensors (CPU) with optional NumPy memmap for large .npy files
    use_numpy = path.endswith('.npy')
    if use_numpy:
        arr = np.load(path, mmap_mode='r')
    else:
        arr: torch.Tensor = torch.load(path, map_location="cpu")
    return arr


def exact_search(
    arr1: np.ndarray | torch.Tensor | str,
    arr2: np.ndarray | torch.Tensor | str,
    k: int,
    metric: str | None = None,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Exact top-K search with chunking to avoid OOM. Supports:
      - metric='ip' (inner product)
      - metric='cosine' (L2-normalized dot-product)
    Backwards-compat: `normalize=True` forces cosine behavior.
    Inputs and outputs are torch-saved (.pt).
    """
    # start_time = time.time()

    if isinstance(arr1, str) and isinstance(arr2, str):
        raise ValueError("NO SUPPORTED")
        arr1 = load_array(arr1)
        arr2 = load_array(arr2)
    
    # Validate input types and shapes; support numpy or torch tensors (or file paths above)
    if not (isinstance(arr1, (torch.Tensor, np.ndarray))):
        raise ValueError("arr1 must be a numpy.ndarray, torch.Tensor, or filepath string")
    if not (isinstance(arr2, (torch.Tensor, np.ndarray))):
        raise ValueError("arr2 must be a numpy.ndarray, torch.Tensor, or filepath string")

    if isinstance(arr1, np.ndarray):
        raise ValueError("Numpy input not supported in this version. Please provide torch.Tensor inputs.")
        assert isinstance(arr2, np.ndarray)
        use_numpy = True
        q_np: np.ndarray = arr1
        d_np: np.ndarray = arr2
        if q_np.ndim != 2 or d_np.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays of shape (N, dim)")
        if q_np.shape[1] != d_np.shape[1]:
            raise ValueError(f"Dim mismatch: queries dim={q_np.shape[1]} vs docs dim={d_np.shape[1]}")
        if q_np.dtype != d_np.dtype:
            raise ValueError(f"Dtype mismatch: queries dtype={q_np.dtype} vs docs dtype={d_np.dtype}")
        Q, dim = q_np.shape
        D, _ = d_np.shape
        # Normalize dtype to supported and record equivalent torch dtype
        if q_np.dtype not in (np.float32, np.float16):
            q_np = q_np.astype(np.float32, copy=False)
            d_np = d_np.astype(np.float32, copy=False)
            tdtype = torch.float32
        else:
            tdtype = torch.float32 if q_np.dtype == np.float32 else torch.float16

    elif isinstance(arr1, torch.Tensor):
        use_numpy = False
        assert isinstance(arr2, torch.Tensor)
        q_tensor: torch.Tensor = arr1
        d_tensor: torch.Tensor = arr2
        
        input_device = q_tensor.device
        if d_tensor.device != input_device:
            raise ValueError("arr1 and arr2 must be on the same device")

        on_gpu_input = (input_device.type == "cuda")

        if q_tensor.dim() != 2 or d_tensor.dim() != 2:
            raise ValueError("Embeddings must be 2D tensors of shape (N, dim)")
        if q_tensor.shape[1] != d_tensor.shape[1]:
            raise ValueError(f"Dim mismatch: queries dim={q_tensor.shape[1]} vs docs dim={d_tensor.shape[1]}")
        if q_tensor.dtype != d_tensor.dtype:
            raise ValueError(f"Dtype mismatch: queries dtype={q_tensor.dtype} vs docs dtype={d_tensor.dtype}")
        if q_tensor.dtype is torch.int8 or d_tensor.dtype is torch.int8:
            raise ValueError("int8 dtype is not supported. Cast inputs to float16, bfloat16, or float32.")
        Q, dim = q_tensor.shape
        D, _ = d_tensor.shape
        tdtype = q_tensor.dtype

    # Validate k
    if k < 1:
        raise ValueError(f"k must be >= 1, got k={k}")
    if k > D:
        raise ValueError(f"k={k} exceeds number of docs D={D}")

    if isinstance(arr1, torch.Tensor):
        torch_device = q_tensor.device
    else:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch_device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        try:
            cpu_threads = int(os.environ.get("TINYKNN_CPU_THREADS", "0"))
            if cpu_threads > 0:
                torch.set_num_threads(cpu_threads)
        except Exception:
            pass


    # Determine operation type
    if metric not in ("ip", "cosine"):
        raise ValueError("metric must be one of {'ip','cosine'}")
    

    emb_bytes = torch.tensor([], dtype=tdtype).element_size()
    score_bytes = 4  # fp32 scores

    # Heuristics
    batch_size = _estimate_batch_size(Q, dim, emb_bytes)
    doc_rows = _estimate_doc_chunk_rows(D, dim, batch_size, emb_bytes, score_bytes)

    empty_cache_mode = os.environ.get("TINYKNN_EMPTY_CACHE", "batch")

    # Outputs on CPU
    topk_scores = torch.empty((Q, k), dtype=torch.float32, device=input_device)
    topk_indices = torch.empty((Q, k), dtype=torch.int64, device=input_device)

    # print(
        # f"Device={torch_device.type}, metric={metric}, Q={Q}x{dim}, D={D}x{dim}, batch_size={batch_size}, doc_rows={doc_rows}, k={k}, dtype={tdtype}")

    with torch.inference_mode():
        stream = torch.cuda.Stream() if torch_device.type == "cuda" and not on_gpu_input else None

        for qs in range(0, Q, batch_size):
            qe = min(qs + batch_size, Q)

            if use_numpy:
                qb = _to_device_chunk(
                    torch.from_numpy(np.asarray(q_np[qs:qe])),
                    torch_device,
                    tdtype,
                )
            else:
                qb = _to_device_chunk(
                    q_tensor[qs:qe],
                    torch_device,
                    tdtype,
                )

            # CPU matmul stability for low precision
            cpu_lowp = (torch_device.type == "cpu" and qb.dtype in (torch.float16, torch.bfloat16))
            if metric == "cosine":
                if cpu_lowp:
                    qb = F.normalize(qb.to(torch.float32), p=2, dim=1, eps=1e-12)
                else:
                    qb = F.normalize(qb, p=2, dim=1, eps=1e-12)

            prev_scores = None
            prev_indices = None

            # Prefetch first chunk
            next_db = None
            if stream:
                with torch.cuda.stream(stream):
                    ds0, de0 = 0, min(doc_rows, D)
                    if use_numpy:
                        db_cpu0 = torch.from_numpy(np.asarray(d_np[ds0:de0]))
                    else:
                        db_cpu0 = d_tensor[ds0:de0]
                    chunk0 = _to_device_chunk(db_cpu0, torch_device, tdtype)
                    if metric == "cosine":
                        if torch_device.type == "cpu" and chunk0.dtype in (torch.float16, torch.bfloat16):
                            chunk0 = F.normalize(chunk0.to(torch.float32), p=2, dim=1, eps=1e-12)
                        else:
                            chunk0 = F.normalize(chunk0, p=2, dim=1, eps=1e-12)
                    next_db = chunk0

            for ds in range(0, D, doc_rows):
                de = min(ds + doc_rows, D)

                if stream:
                    torch.cuda.current_stream().wait_stream(stream)
                    db = next_db
                    # Prefetch next
                    ns = ds + doc_rows
                    if ns < D:
                        with torch.cuda.stream(stream):
                            ne = min(ns + doc_rows, D)
                            if use_numpy:
                                db_cpun = torch.from_numpy(np.asarray(d_np[ns:ne]))
                            else:
                                db_cpun = d_tensor[ns:ne]
                            chunk = _to_device_chunk(db_cpun, torch_device, tdtype)
                            if metric == "cosine":
                                if torch_device.type == "cpu" and chunk.dtype in (torch.float16, torch.bfloat16):
                                    chunk = F.normalize(chunk.to(torch.float32), p=2, dim=1, eps=1e-12)
                                else:
                                    chunk = F.normalize(chunk, p=2, dim=1, eps=1e-12)
                            next_db = chunk
                else:
                    if use_numpy:
                        db_cpu = torch.from_numpy(np.asarray(d_np[ds:de]))
                    else:
                        db_cpu = d_tensor[ds:de]
                    db = _to_device_chunk(db_cpu, torch_device, tdtype)
                    if metric == "cosine":
                        if torch_device.type == "cpu" and db.dtype in (torch.float16, torch.bfloat16):
                            db = F.normalize(db.to(torch.float32), p=2, dim=1, eps=1e-12)
                        else:
                            db = F.normalize(db, p=2, dim=1, eps=1e-12)

                # Matmul with proper numerics
                if cpu_lowp:
                    scores = torch.matmul(qb.to(torch.float32), db.to(torch.float32).t())
                elif torch_device.type == "cuda" and qb.dtype in (torch.float16, torch.bfloat16):
                    # Use modern autocast API; GEMM uses FP16/BF16 inputs with FP32 accumulate
                    with torch.amp.autocast(device_type='cuda', dtype=qb.dtype):
                        scores = torch.matmul(qb, db.t())
                else:
                    scores = torch.matmul(qb, db.t())

                chunk_k = min(k, de - ds)
                vals, idx = torch.topk(scores.float(), k=chunk_k, dim=1, largest=True, sorted=True)
                idx = idx + ds

                if prev_scores is None:
                    prev_scores, prev_indices = vals, idx
                else:
                    combined_scores = torch.cat([prev_scores, vals], dim=1)
                    combined_indices = torch.cat([prev_indices, idx], dim=1)
                    if combined_scores.shape[1] > k:
                        prev_scores, pick = torch.topk(combined_scores, k=k, dim=1, largest=True, sorted=True)
                        prev_indices = torch.gather(combined_indices, 1, pick)
                    else:
                        prev_scores, prev_indices = combined_scores, combined_indices

                del db, scores, vals, idx
                if torch_device.type == "cuda" and empty_cache_mode == "chunk":
                    torch.cuda.empty_cache()
            if not on_gpu_input:
                topk_scores[qs:qe] = prev_scores.cpu()
                topk_indices[qs:qe] = prev_indices.cpu()
            else:
                topk_scores[qs:qe] = prev_scores
                topk_indices[qs:qe] = prev_indices
                
            del qb, prev_scores, prev_indices
            if torch_device.type == "cuda" and not on_gpu_input and empty_cache_mode in ("batch", "chunk"):
                torch.cuda.empty_cache()

            pct = 100.0 * min(qe, Q) / max(Q, 1)
            # print(f"Processed queries {qs}:{qe} ({pct:.1f}%)")

    # elapsed = time.time() - start_time
    # print(f"Done in {elapsed:.2f}s")
    if use_numpy:
        return topk_indices.numpy(), topk_scores.numpy()
    return topk_indices, topk_scores