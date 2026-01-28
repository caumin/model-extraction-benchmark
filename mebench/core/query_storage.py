"""Query data storage for Track A training caching."""

from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
import pickle
import shutil


class QueryStorage(Dataset):
    """Storage for queried data (x, y) tensors for Track A training.

    This provides:
    1. Batch-level append during oracle queries
    2. Random access for DataLoader during training
    3. Automatic cleanup on completion
    4. Run-scoped directory structure

    Storage format:
    - queries.pt: Concatenated tensor [N, C, H, W] of all images
    - labels.pt: Tensor [N] of oracle outputs (labels or probs)

    For soft_prob mode: labels stores probability distributions [N, K]
    For hard_top1 mode: labels stores class indices [N]
    """

    def __init__(self, cache_dir: Path, output_mode: str = "soft_prob"):
        """Initialize query storage.

        Args:
            cache_dir: Directory for storage files
            output_mode: Oracle output mode (soft_prob or hard_top1)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_mode = output_mode

        # Storage buffers (list of tensors)
        self.query_chunks: list[torch.Tensor] = []
        self.label_chunks: list[torch.Tensor] = []
        self.chunk_sizes: list[int] = []
        self.count = 0
        
        # Cumulative indices for efficient random access
        self.cumulative_sizes: list[int] = [0]

    def add_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
        """Append a batch of queries to storage.

        Args:
            x_batch: Query images [B, C, H, W]
            y_batch: Oracle outputs [B] or [B, K]
        """
        # Move to CPU for storage
        x_cpu = x_batch.detach().cpu()
        y_cpu = y_batch.detach().cpu()
        
        batch_size = x_cpu.shape[0]
        if batch_size == 0:
            return

        self.query_chunks.append(x_cpu)
        self.label_chunks.append(y_cpu)
        self.chunk_sizes.append(batch_size)
        self.count += batch_size
        self.cumulative_sizes.append(self.count)

    def save(self) -> None:
        """Save current storage to disk.
        
        Currently DISABLED to save disk space.
        
        Note: Currently saves all chunks to a single file. 
        TODO: Implement incremental append or chunk-based saving for very large datasets.
        """
        # Save is disabled for now as it consumes too much disk space
        return

        if self.count == 0:
            return

        queries_path = self.cache_dir / "queries.pt"
        labels_path = self.cache_dir / "labels.pt"
        meta_path = self.cache_dir / "meta.pkl"

        # Concatenate for saving (or save list?)
        # Concatenating once at save time is better than concatenating every batch.
        all_queries = torch.cat(self.query_chunks, dim=0)
        all_labels = torch.cat(self.label_chunks, dim=0)

        # Save tensors
        torch.save(all_queries, queries_path)
        torch.save(all_labels, labels_path)

        # Save metadata
        meta = {
            "count": self.count,
            "output_mode": self.output_mode,
            "queries_shape": all_queries.shape,
            "labels_shape": all_labels.shape,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        print(f"Saved {self.count} queries to {self.cache_dir}")

    def load(self) -> None:
        """Load storage from disk."""
        queries_path = self.cache_dir / "queries.pt"
        labels_path = self.cache_dir / "labels.pt"
        meta_path = self.cache_dir / "meta.pkl"

        if not queries_path.exists() or not labels_path.exists():
            return

        # Load tensors
        queries = torch.load(queries_path, weights_only=True)
        labels = torch.load(labels_path, weights_only=True)
        
        # Reset buffers and load as a single chunk
        self.query_chunks = [queries]
        self.label_chunks = [labels]
        self.chunk_sizes = [queries.shape[0]]
        self.count = queries.shape[0]
        self.cumulative_sizes = [0, self.count]

        # Load metadata
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.output_mode = meta.get("output_mode", self.output_mode)

        print(f"Loaded {self.count} queries from {self.cache_dir}")

    def get_dataloader(
        self, batch_size: int = 128, shuffle: bool = True, num_workers: int = 0
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader for stored queries.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of workers

        Returns:
            DataLoader
        """
        if self.count == 0:
            # Return empty loader if no data
            return torch.utils.data.DataLoader([])
            
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    def cleanup(self) -> None:
        """Remove all storage files."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print(f"Cleaned up cache at {self.cache_dir}")

    def __len__(self) -> int:
        """Return number of stored queries."""
        return self.count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single query by index.

        Efficiently locates the correct chunk and index within that chunk.

        Args:
            idx: Query index

        Returns:
            (x, y) tuple
        """
        if self.count == 0:
             raise IndexError("QueryStorage is empty")
             
        if idx < 0:
            idx += self.count
        if idx >= self.count or idx < 0:
            raise IndexError(f"Index {idx} out of range [0, {self.count-1}]")

        # Binary search for the chunk containing idx
        # self.cumulative_sizes stores [0, s1, s1+s2, ...]
        # We want i such that cumulative_sizes[i] <= idx < cumulative_sizes[i+1]
        
        # Bisect right gives index where idx would be inserted. 
        # Since idx is 0-based index of item, and cum_sizes are accumulated counts:
        # Example: chunks=[10, 20], cum=[0, 10, 30]. idx=5 -> chunk 0. idx=15 -> chunk 1.
        
        import bisect
        chunk_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        
        offset = idx - self.cumulative_sizes[chunk_idx]
        return self.query_chunks[chunk_idx][offset], self.label_chunks[chunk_idx][offset]


def create_query_storage(
    run_dir: Path,
    output_mode: str = "soft_prob",
) -> QueryStorage:
    """Create query storage in run directory.

    Args:
        run_dir: Run directory path
        output_mode: Oracle output mode

    Returns:
        QueryStorage instance
    """
    cache_dir = run_dir / "query_cache"
    return QueryStorage(cache_dir, output_mode)
