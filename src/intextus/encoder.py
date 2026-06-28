import os
from typing import List, Union
import numpy as np

# Import the C++ class under an alias
from ._core import IntextusEncoder as CppIntextusEncoder

class DenseEncoder:
    def __init__(
        self, 
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", 
        tokenizer_path: str = None, 
        do_lower_case: bool = None,
        num_threads: int = 0,
        cls_token_id: int = None,
        sep_token_id: int = None,
        quantization: str = "Q8_0",
        pooling_mode: str = None
    ):
        """
        Inference engine for running dense embedding GGUF models 
        accelerated via llama.cpp.
        
        Args:
            model_name_or_path: Local path to a GGUF file/directory, or a Hugging Face Hub model ID.
            tokenizer_path: Optional path to tokenizer.json. If None, it is resolved automatically.
            do_lower_case: Whether to lower case input texts. If None, determined automatically.
            num_threads: Number of threads for parallel execution.
            cls_token_id: Optional exact token ID for CLS.
            sep_token_id: Optional exact token ID for SEP.
            quantization: Preferred quantization format (e.g. "Q8_0", "F32", "Q4_K_M"). Defaults to "Q8_0".
        """
        gguf_path = None
        
        if os.path.exists(model_name_or_path):
            if os.path.isdir(model_name_or_path):
                # Search for a GGUF file in the local directory
                import glob
                gguf_files = glob.glob(os.path.join(model_name_or_path, "*.gguf"))
                if gguf_files:
                    # Filter by specified quantization if possible
                    selected = None
                    if quantization:
                        for f in gguf_files:
                            if quantization.upper() in os.path.basename(f).upper():
                                selected = f
                                break
                    if selected is None:
                        # Fallback to Q8_0
                        for f in gguf_files:
                            if "Q8_0" in os.path.basename(f):
                                selected = f
                                break
                    if selected is None:
                        selected = gguf_files[0]
                    gguf_path = selected
                else:
                    raise FileNotFoundError(f"No GGUF file (*.gguf) found in directory {model_name_or_path}")
                
                if tokenizer_path is None:
                    tokenizer_path = os.path.join(model_name_or_path, "tokenizer.json")
            else:
                if model_name_or_path.endswith(".gguf"):
                    gguf_path = model_name_or_path
                    if tokenizer_path is None:
                        tokenizer_path = os.path.join(os.path.dirname(model_name_or_path), "tokenizer.json")
                else:
                    raise ValueError(f"Local path must be a directory containing a GGUF file or direct path to a .gguf file: {model_name_or_path}")
        else:
            repo_id = model_name_or_path
            if repo_id == "sentence-transformers/all-MiniLM-L6-v2":
                repo_id = "intextus/all-MiniLM-L6-v2-GGUF"
            elif repo_id == "BAAI/bge-small-en-v1.5":
                repo_id = "intextus/bge-small-en-v1.5-GGUF"
                
            try:
                from huggingface_hub import HfApi, hf_hub_download
                print(f"Downloading GGUF assets from Hugging Face repository '{repo_id}'...")
                
                # Fetch tokenizer.json
                if tokenizer_path is None:
                    try:
                        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
                    except Exception:
                        print("tokenizer.json not found in GGUF repository. Falling back to sentence-transformers/all-MiniLM-L6-v2 tokenizer...")
                        tokenizer_path = hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="tokenizer.json")
                
                # Dynamically look for any .gguf file in the repository
                api = HfApi()
                files = api.list_repo_files(repo_id=repo_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                if not gguf_files:
                    raise FileNotFoundError(f"No .gguf file found in Hugging Face repository '{repo_id}'")
                
                # Prefer the requested quantization format if available
                gguf_filename = None
                if quantization:
                    for f in gguf_files:
                        if quantization.upper() in f.upper():
                            gguf_filename = f
                            break
                # Fallback to Q8_0 if requested was not found
                if gguf_filename is None:
                    for f in gguf_files:
                        if "Q8_0" in f:
                            gguf_filename = f
                            break
                # Fallback to any GGUF file
                if gguf_filename is None:
                    gguf_filename = gguf_files[0]
                    
                gguf_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)
            except Exception as e:
                raise ValueError(
                    f"Could not load GGUF model '{model_name_or_path}' from local path or Hugging Face Hub.\n"
                    f"Underlying error: {e}"
                )
                
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF model file not found at {gguf_path}")
            
        if tokenizer_path is None or not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        # Resolve do_lower_case from tokenizer_config.json if not specified
        if do_lower_case is None:
            config_do_lower = None
            if tokenizer_path:
                config_path = os.path.join(os.path.dirname(tokenizer_path), "tokenizer_config.json")
                if os.path.exists(config_path):
                    try:
                        import json
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                        config_do_lower = config_data.get("do_lower_case")
                    except Exception:
                        pass
            if config_do_lower is not None:
                do_lower_case = config_do_lower
            else:
                do_lower_case = "uncased" in model_name_or_path.lower()

        # Determine token IDs dynamically if not provided
        vocab_tokens = {}
        try:
            import json
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tok_data = json.load(f)
            vocab = tok_data.get("model", {}).get("vocab", {})
            if vocab:
                if isinstance(vocab, list):
                    vocab_tokens = {item[0] if isinstance(item, list) else item: i for i, item in enumerate(vocab)}
                else:
                    vocab_tokens = vocab
        except Exception:
            pass

        # Try to find standard CLS/SEP IDs
        default_cls_id = vocab_tokens.get("[CLS]", vocab_tokens.get("<s>", 101))
        default_sep_id = vocab_tokens.get("[SEP]", vocab_tokens.get("</s>", 102))

        cls_token_id = cls_token_id if cls_token_id is not None else default_cls_id
        sep_token_id = sep_token_id if sep_token_id is not None else default_sep_id

        # Determine pooling mode and type
        if pooling_mode is None:
            if "bge" in model_name_or_path.lower():
                pooling_mode = "cls"
            else:
                pooling_mode = "mean"
        
        pooling_mode = pooling_mode.lower()
        if pooling_mode == "cls":
            pooling_type = 1
        elif pooling_mode == "mean":
            pooling_type = 0
        else:
            raise ValueError(f"Unsupported pooling_mode: {pooling_mode}. Must be 'cls' or 'mean'.")

        # Read and modify tokenizer.json to disable padding if configured
        self._temp_tokenizer_file = None
        try:
            import json
            import tempfile
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tok_data = json.load(f)
            if tok_data.get("padding") is not None:
                tok_data["padding"] = None
                fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="intextus_tokenizer_")
                with os.fdopen(fd, 'w', encoding='utf-8') as f_out:
                    json.dump(tok_data, f_out)
                self._temp_tokenizer_file = temp_path
                tokenizer_path = temp_path
        except Exception:
            pass

        # Initialize C++ core encoder with direct GGUF and tokenizer path
        self._encoder = CppIntextusEncoder(
            gguf_path,
            tokenizer_path,
            do_lower_case,
            num_threads,
            cls_token_id,
            sep_token_id,
            pooling_type
        )

    def __del__(self):
        if getattr(self, "_temp_tokenizer_file", None) and os.path.exists(self._temp_tokenizer_file):
            try:
                os.remove(self._temp_tokenizer_file)
            except Exception:
                pass

    def encode(self, texts: Union[str, List[str]], max_length: int = 512, normalize: bool = True) -> np.ndarray:
        """
        Encodes input texts into dense embeddings using the accelerated C++ backend.
        
        Args:
            texts: A single text string or list of text strings.
            max_length: Maximum sequence length.
            normalize: Whether to apply L2 normalization to the output vectors.
            
        Returns:
            A 2D NumPy array of shape (Batch, Dim).
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            # We determine output dimension dynamically, but default to 384 or query C++
            dim = getattr(self._encoder, "model_embd_dim_", 384)
            return np.zeros((0, dim), dtype=np.float32)
        return self._encoder.encode(texts, max_length, normalize)
