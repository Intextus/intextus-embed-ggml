# intextus

[![PyPI Version](https://img.shields.io/pypi/v/intextus-embed-ggml.svg)](https://pypi.org/project/intextus-embed-ggml/)
[![CI/CD Status](https://github.com/intextus/intextus-embed-ggml/actions/workflows/publish.yml/badge.svg)](https://github.com/intextus/intextus-embed-ggml/actions/workflows/publish.yml)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/intextus-embed-ggml)](https://pypi.org/project/intextus-embed-ggml/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://pypi.org/project/intextus-embed-ggml/)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-blue.svg)](https://pypi.org/project/intextus-embed-ggml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, zero-PyTorch, zero-ONNX-Runtime dense embedding inference engine. Uses a native C++ extension (`llama.cpp` + `tokenizers-cpp`) to encode text extremely fast without pulling in gigabytes of deep learning dependencies.

## Install

```bash
pip install intextus-embed-ggml
```

Only runtime dependencies are `numpy` and `huggingface-hub`.

## Usage

```python
from intextus import DenseEncoder

# Initializes the encoder, downloading the optimized Q8_0 quantized all-MiniLM-L6-v2 GGUF model automatically
model = DenseEncoder("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(["What is a dense embedding?", "It's extremely fast and lightweight."])
print(embeddings.shape)  # (2, 384)

# Load BAAI General Embedding model (auto-resolves to GGUF and uses CLS pooling)
bge_model = DenseEncoder("BAAI/bge-small-en-v1.5")
bge_embeddings = bge_model.encode(["BAAI General Embedding models use CLS pooling."])
```

You can also point it at a local directory containing a `.gguf` file and `tokenizer.json` or directly to a `.gguf` file path:

```python
model = DenseEncoder("./my-local-model-directory/")
# OR
model = DenseEncoder("./models/all-MiniLM-L6-v2-Q8_0.gguf")
```

## Configuration Options

- `model_name_or_path`: Local path to a GGUF file or directory, or a Hugging Face Hub model ID (defaults to `"sentence-transformers/all-MiniLM-L6-v2"`).
- `tokenizer_path`: Optional explicit path to `tokenizer.json`.
- `num_threads`: Number of CPU threads to use. Defaults to `0` (which automatically detects and uses physical CPU cores, avoiding hyperthreading bottlenecks).
- `quantization`: Preferred quantization format (e.g., `"Q8_0"`, `"F16"`, `"F32"`, `"Q4_0"`). Defaults to `"Q8_0"`.
- `pooling_mode`: Pooling strategy to use (`"mean"` or `"cls"`). Defaults to `None` (which auto-detects based on the model name).

## Advanced: Compile from Source (Hardware Acceleration)

By default, pre-built binary wheels are compiled with native SIMD instructions (AVX2/AVX-512/ARM NEON) for maximum CPU portability. If you are compiling from source and want to link against optimized system BLAS backends, pass the appropriate CMake arguments during installation:

- **AMD / Generic CPUs (OpenBLAS)**:
  ```bash
  CMAKE_ARGS="-DGGML_OPENBLAS=ON" pip install --no-binary :all: intextus-embed-ggml
  ```
- **Intel CPUs (Intel MKL / oneDNN)**:
  ```bash
  CMAKE_ARGS="-DGGML_MKL=ON" pip install --no-binary :all: intextus-embed-ggml
  ```

## Features

- **GGUF-Native**: Avoids PyTorch and ONNX Runtime entirely.
- **Hardware Optimized**: Compiled with native SIMD instructions (AVX2/AVX-512/ARM NEON) and Flash Attention support.
- **Dynamic Threading**: Auto-detects physical CPU cores to prevent runtime CPU thread thrashing.
- **Highly Portable**: No complex system level dependencies, builds easily on macOS, Linux, and Windows.

## License

MIT. See [LICENSE](LICENSE).
