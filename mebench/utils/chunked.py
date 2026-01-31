"""Memory-efficient utilities for chunked processing of large datasets."""

import torch
from typing import Iterator, List, Callable, Any, Optional
import logging


class ChunkedProcessor:
    """Memory-efficient chunked processing for large tensors and datasets.
    
    This utility helps process large datasets that don't fit in memory by:
    1. Processing data in configurable chunk sizes
    2. Using streaming operations instead of materializing full tensors
    3. Providing progress tracking for long operations
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize chunked processor.
        
        Args:
            chunk_size: Number of items to process at once
            device: Device to move chunks to (if specified)
            verbose: Whether to log progress information
        """
        self.chunk_size = chunk_size
        self.device = device
        self.verbose = verbose
        
    def process_tensor_chunks(
        self,
        tensor: torch.Tensor,
        process_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int = 0
    ) -> torch.Tensor:
        """Process large tensor in chunks along specified dimension.
        
        Args:
            tensor: Input tensor to process
            process_fn: Function to apply to each chunk
            dim: Dimension along which to chunk (default: 0)
            
        Returns:
            Concatenated result of processing all chunks
        """
        total_size = tensor.shape[dim]
        results = []
        
        if self.verbose:
            logging.info(f"Processing tensor of shape {tensor.shape} in chunks of {self.chunk_size}")
        
        for i in range(0, total_size, self.chunk_size):
            end_idx = min(i + self.chunk_size, total_size)
            
            # Extract chunk
            if dim == 0:
                chunk = tensor[i:end_idx]
            else:
                # Use slice for other dimensions
                slices = [slice(None)] * len(tensor.shape)
                slices[dim] = slice(i, end_idx)
                chunk = tensor[slices]
            
            # Move to device if specified
            if self.device is not None:
                chunk = chunk.to(self.device)
            
            # Process chunk
            processed_chunk = process_fn(chunk)
            results.append(processed_chunk)
            
            if self.verbose:
                logging.info(f"Processed chunk {i//self.chunk_size + 1}/{(total_size-1)//self.chunk_size + 1}")
        
        # Concatenate results
        if results:
            return torch.cat(results, dim=dim)
        else:
            # Return empty tensor with same shape except processed dimension
            shape = list(tensor.shape)
            shape[dim] = 0
            return torch.empty(shape, dtype=results[0].dtype if results else tensor.dtype)
    
    def stream_dataset_items(
        self,
        dataset,
        process_fn: Callable[[Any], None],
        max_items: Optional[int] = None
    ) -> None:
        """Stream dataset items in chunks without materializing full dataset.
        
        Args:
            dataset: Dataset-like object with __len__ and __getitem__
            process_fn: Function to call on each item
            max_items: Maximum number of items to process (None for all)
        """
        total_items = min(len(dataset), max_items) if max_items else len(dataset)
        
        if self.verbose:
            logging.info(f"Streaming {total_items} items in chunks of {self.chunk_size}")
        
        for i in range(0, total_items, self.chunk_size):
            end_idx = min(i + self.chunk_size, total_items)
            
            for j in range(i, end_idx):
                item = dataset[j]
                process_fn(item)
            
            if self.verbose:
                logging.info(f"Processed items {i}-{end_idx-1}/{total_items}")
    
    def accumulate_chunks(
        self,
        data_loader,
        process_fn: Callable[[torch.Tensor], torch.Tensor],
        accumulate_fn: Callable[[List[torch.Tensor]], torch.Tensor]
    ) -> torch.Tensor:
        """Accumulate processed chunks from data loader without memory explosion.
        
        Args:
            data_loader: DataLoader providing batches
            process_fn: Function to process each batch
            accumulate_fn: Function to combine all processed batches
            
        Returns:
            Accumulated result
        """
        processed_chunks = []
        
        if self.verbose:
            logging.info(f"Accumulating chunks from {len(data_loader)} batches")
        
        for i, batch in enumerate(data_loader):
            # Move to device if specified
            if self.device is not None:
                batch = batch.to(self.device) if torch.is_tensor(batch) else tuple(
                    x.to(self.device) if torch.is_tensor(x) else x for x in batch
                )
            
            # Process batch
            processed = process_fn(batch)
            processed_chunks.append(processed)
            
            if self.verbose and (i + 1) % 10 == 0:
                logging.info(f"Processed {i + 1}/{len(data_loader)} batches")
        
        # Accumulate all chunks
        if processed_chunks:
            result = accumulate_fn(processed_chunks)
            
            if self.verbose:
                logging.info(f"Accumulated {len(processed_chunks)} chunks into result of shape {result.shape}")
            
            return result
        else:
            raise ValueError("No chunks to accumulate")


def create_chunked_dataloader(
    dataset,
    batch_size: int,
    chunk_size: int = 1000,
    shuffle: bool = True,
    num_workers: int = 0,
    device: Optional[str] = None
):
    """Create a chunked data loader for memory-efficient processing.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size for individual batches
        chunk_size: Number of samples to load at once
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        device: Device to move batches to
        
    Yields:
        Tuple of (batch, chunk_info) for each chunk
    """
    from torch.utils.data import DataLoader, Subset
    
    total_size = len(dataset)
    
    # Create indices
    indices = list(range(total_size))
    if shuffle:
        import random
        random.shuffle(indices)
    
    # Process in chunks
    for chunk_start in range(0, total_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_size)
        chunk_indices = indices[chunk_start:chunk_end]
        
        # Create subset for this chunk
        chunk_dataset = Subset(dataset, chunk_indices)
        
        # Create data loader for this chunk
        chunk_loader = DataLoader(
            chunk_dataset,
            batch_size=batch_size,
            shuffle=False,  # Already shuffled
            num_workers=num_workers,
            drop_last=False
        )
        
        chunk_info = {
            "chunk_id": chunk_start // chunk_size,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "chunk_size": len(chunk_indices),
            "total_chunks": (total_size - 1) // chunk_size + 1
        }
        
        yield chunk_loader, chunk_info


def memory_efficient_cat(
    tensors: List[torch.Tensor],
    dim: int = 0,
    max_memory_mb: int = 1000
) -> torch.Tensor:
    """Memory-efficient concatenation that avoids overflow for large tensors.
    
    Args:
        tensors: List of tensors to concatenate
        dim: Dimension along which to concatenate
        max_memory_mb: Maximum memory to use for intermediate results
        
    Returns:
        Concatenated tensor
    """
    if not tensors:
        raise ValueError("Cannot concatenate empty list of tensors")
    
    if len(tensors) == 1:
        return tensors[0]
    
    # Estimate memory usage
    total_elements = sum(t.numel() for t in tensors)
    dtype_bytes = tensors[0].element_size()
    estimated_mb = (total_elements * dtype_bytes) / (1024 * 1024)
    
    if estimated_mb <= max_memory_mb:
        # Safe to concatenate all at once
        return torch.cat(tensors, dim=dim)
    
    # Use tree-like concatenation to reduce peak memory
    if len(tensors) <= 2:
        return torch.cat(tensors, dim=dim)
    
    # Split and concatenate recursively
    mid = len(tensors) // 2
    left = memory_efficient_cat(tensors[:mid], dim, max_memory_mb)
    right = memory_efficient_cat(tensors[mid:], dim, max_memory_mb)
    
    return torch.cat([left, right], dim=dim)


def chunked_inference(
    model,
    data_loader,
    device: str,
    chunk_size: int = 1000,
    max_memory_mb: int = 1000
) -> torch.Tensor:
    """Perform memory-efficient inference on large datasets.
    
    Args:
        model: Neural network model for inference
        data_loader: DataLoader providing input batches
        device: Device to run inference on
        chunk_size: Number of samples to process at once
        max_memory_mb: Maximum memory for storing results
        
    Returns:
        Concatenated model outputs
    """
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                x_batch = batch[0].to(device)
            else:
                x_batch = batch.to(device)
            
            # Forward pass
            batch_output = model(x_batch)
            outputs.append(batch_output.cpu())
            
            # Periodically concatenate to manage memory
            if len(outputs) >= chunk_size // data_loader.batch_size:
                if len(outputs) > 1:
                    outputs = [memory_efficient_cat(outputs, dim=0, max_memory_mb=max_memory_mb)]
        
        # Final concatenation
        if outputs:
            return memory_efficient_cat(outputs, dim=0, max_memory_mb=max_memory_mb)
        else:
            # Return empty tensor with correct shape
            return torch.empty((0, *batch_output.shape[1:]))
    
    return memory_efficient_cat(outputs, dim=0, max_memory_mb=max_memory_mb)