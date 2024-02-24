# Some words:

This repo consists almost full torch code of LRU Layer from DeepMinds recent paper. However, due to limitations of nowadays PyTorch framework there are some opperations that require JAX and the following library jax2torch that tracks JAX gradients and converts them into PyTorch gradients. I have not checked how jax2torch library slows forward-backward pass, planning to do so in the nearest future. Although, this is just a note, if you are willing to use this layer - do not bother yourself of thinking about jax, just install it and forget it will work with torch tensors without any magic.



Before using you'll need to perform the code below.

```bash
pip install jax
pip install jax2torch
```

## Usage:

```python
from LRU import LRULayer

lru = LRULayer(*args, **kwargs)
```




