# Pytorch's Advanced Knowledge

## Reproducibility

> https://pytorch.org/docs/stable/notes/randomness.html

### Pytorch random number generator

Seed the RNG(random number generator) for all devices(both CPU and CUDA)

```python
torch.manual_seed(233)
torch.cuda.manual_seed_all(233)  # Set the seed for all gpus
```

### Python

```python
random.seed(233)
```

### Other libraries

```python
np.random.seed(233)
```

### CUDA convolution benchmark

When a cuDNN convolution is called with a new set of size parameters, pytorch will run multiple convolution algorithms and `benchmark` them to find the fastest one. Then, the `fastest` algorithm will be used consistently during the rest of the process for the corresponding set of size parameters.

However, disabling the benchmark feature with `torch.backbends.cudnn.benchmark = False` causes cuDNN to deterministically select an convolution algorithm, possibly at the cost of reduced performance.

### CUDA determinism

`torch.use_deterministic_algorithms(True)` will make all pytorch operations behave deterministically, while `torch.backends.cudnn.deterministic = True` will control the `torch.backbends.cudnn.benchmark = False` itself to be deterministic.

### DataLoader

DataLoader will reseed workers, which are new process. Use `worker_init_fn()` to preserve reproducibility.

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=seed_worker
)
```

## Resume

During training:

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optim_state_dict': optimizer.state_dict(),
}
```

How to resume:

```python
def warmup_model_data(num_epochs, data_loader, model):
    for epoch in range(num_epochs):
        for images in data_loader:
            with torch.no_grad():
                # If and only if there are Dropout layers in model
                # BatchNorm layers should be set to 'eval' mode
                model(images)
epoch = checkpoint['epoch'] + 1
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optim_state_dict'])
warmup_model_data(epoch, data_loader, model)
```

## `pin_memory` and `non_blocking`

> https://spell.ml/blog/pytorch-training-tricks-YAnJqBEAACkARhgD

A **page** is an automic unit of memory in OS-level **virtual memory** management. A standard page size is 4KB; larger page sizes are possible for modern OSes and systems.

**Pagging** is the act of stroing and retrieving memory pages from disk to main memory. Paging is used to allow the working memory set of the application running on the OS to exceed the total RAM.

All memory is managed in pages, but pagging is only used when the working set spills to disk.

In CUDA, RAM is referred as **pinned memory**. Pinning a block of memory can be done via a CUDA API call, which issues an OS call that reserves the memory block and sets the constraint that it cannot be spilled to disk.

Pinned memory is used to speed up a CPU to GPU memory copy operation (as executed by e.g. tensor.cuda() in PyTorch) by ensuring that none of the memory that is to be copied is on disk.

Setting `num_workers` to a value other than its default `0` will spin up worker processes that individually load and transform the data (e.g. multiprocessing) and load it into the main memory of the host process.

Non-Blocking allows you to overlap compute and memory transfer to the GPU. The reason you can set the target as non-blocking is so you can overlap the compute of the model and the transfer of the ground-truth. If you set the input to also non-blocking, it would yield no benefit due to the fact that the model has a dependency on the input data.

Pinned Memory allows the non-blocking calls to actually be non-blocking.