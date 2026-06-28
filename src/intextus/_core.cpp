#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <llama.h>
#include <tokenizers_cpp.h>

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <cctype>
#include <cstring>
#include <thread>

namespace nb = nanobind;

static std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::binary | std::ios::ate);
    if (!fs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    auto size = fs.tellg();
    fs.seekg(0, std::ios::beg);
    std::string buf(size, '\0');
    fs.read(buf.data(), size);
    return buf;
}

class IntextusEncoder {
public:
    IntextusEncoder(
        const std::string& gguf_path,
        const std::string& tokenizer_path,
        bool do_lower_case,
        int num_threads,
        int cls_token_id,
        int sep_token_id
    ) : do_lower_case_(do_lower_case),
        cls_token_id_(cls_token_id),
        sep_token_id_(sep_token_id) {

        // Load tokenizer
        std::string tok_blob = LoadBytesFromFile(tokenizer_path);
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tok_blob);

        // Load model
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        llama_model_ = llama_load_model_from_file(gguf_path.c_str(), model_params);
        if (!llama_model_) {
            throw std::runtime_error("Failed to load GGUF model: " + gguf_path);
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.embeddings = true;
        int actual_threads = num_threads;
        if (actual_threads <= 0) {
            actual_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);
        }
        ctx_params.n_threads = actual_threads;
        ctx_params.n_threads_batch = actual_threads;
        ctx_params.n_ctx = 512;
        ctx_params.n_batch = 512;
        ctx_params.flash_attn = true;
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;

        llama_ctx_ = llama_new_context_with_model(llama_model_, ctx_params);
        if (!llama_ctx_) {
            throw std::runtime_error("Failed to create llama.cpp context");
        }

        model_embd_dim_ = llama_n_embd(llama_model_);
    }

    ~IntextusEncoder() {
        if (llama_ctx_) {
            llama_free(llama_ctx_);
        }
        if (llama_model_) {
            llama_free_model(llama_model_);
        }
    }

    nb::ndarray<float, nb::numpy, nb::device::cpu> encode(
        const std::vector<std::string>& texts,
        size_t max_length,
        bool normalize
    ) {
        if (texts.empty()) {
            float* empty_data = new float[0];
            size_t shape[2] = {0, 0};
            nb::capsule owner(empty_data, [](void* p) noexcept {
                delete[] static_cast<float*>(p);
            });
            return nb::ndarray<float, nb::numpy, nb::device::cpu>(empty_data, 2, shape, owner);
        }

        size_t batch_size = texts.size();
        size_t output_dim = model_embd_dim_;
        float* result_data = new float[batch_size * output_dim];
        std::memset(result_data, 0, batch_size * output_dim * sizeof(float));

        {
            nb::gil_scoped_release release;

            for (size_t b = 0; b < batch_size; ++b) {
                const std::string* text_ptr = &texts[b];
                std::string lower_buf;
                if (do_lower_case_) {
                    lower_buf = texts[b];
                    std::transform(lower_buf.begin(), lower_buf.end(), lower_buf.begin(),
                        [](unsigned char c) { return std::tolower(c); });
                    text_ptr = &lower_buf;
                }

                std::vector<int> raw_ids = tokenizer_->Encode(*text_ptr);

                // Setup sequence with [CLS] and [SEP]
                std::vector<int64_t> seq;
                seq.push_back(cls_token_id_);
                for (int id : raw_ids) {
                    if (id != cls_token_id_ && id != sep_token_id_) {
                        seq.push_back(id);
                    }
                }
                
                // Truncate if exceeding bounds (leaving room for [SEP])
                if (seq.size() >= max_length) {
                    seq.resize(max_length - 1);
                }
                seq.push_back(sep_token_id_);

                size_t decode_len = seq.size();

                llama_kv_cache_clear(llama_ctx_);
                llama_batch batch = llama_batch_init(decode_len, 0, 1);
                batch.n_tokens = decode_len;

                for (size_t s = 0; s < decode_len; ++s) {
                    batch.token[s] = seq[s];
                    batch.pos[s] = s;
                    batch.n_seq_id[s] = 1;
                    batch.seq_id[s][0] = 0;
                    batch.logits[s] = true;
                }

                if (llama_decode(llama_ctx_, batch) != 0) {
                    llama_batch_free(batch);
                    delete[] result_data;
                    throw std::runtime_error("llama_decode failed");
                }

                // Mean Pooling over decoded token embeddings
                float* out_vec = result_data + b * output_dim;
                for (size_t s = 0; s < decode_len; ++s) {
                    const float * embd = llama_get_embeddings_ith(llama_ctx_, s);
                    if (!embd) {
                        llama_batch_free(batch);
                        delete[] result_data;
                        throw std::runtime_error("Failed to get embeddings for token");
                    }
                    for (size_t k = 0; k < output_dim; ++k) {
                        out_vec[k] += embd[k];
                    }
                }

                float scale = 1.0f / static_cast<float>(decode_len);
                for (size_t k = 0; k < output_dim; ++k) {
                    out_vec[k] *= scale;
                }

                // L2 Normalization
                if (normalize) {
                    float norm_sq = 0.0f;
                    for (size_t k = 0; k < output_dim; ++k) {
                        norm_sq += out_vec[k] * out_vec[k];
                    }
                    float norm = std::sqrt(norm_sq);
                    float norm_scale = norm > 1e-12f ? 1.0f / norm : 0.0f;
                    for (size_t k = 0; k < output_dim; ++k) {
                        out_vec[k] *= norm_scale;
                    }
                }

                llama_batch_free(batch);
            }
        }

        size_t shape[2] = {batch_size, output_dim};
        nb::capsule owner(result_data, [](void* p) noexcept {
            delete[] static_cast<float*>(p);
        });
        return nb::ndarray<float, nb::numpy, nb::device::cpu>(result_data, 2, shape, owner);
    }

private:
    struct llama_model * llama_model_ = nullptr;
    struct llama_context * llama_ctx_ = nullptr;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

    bool do_lower_case_;
    int cls_token_id_ = -1;
    int sep_token_id_ = -1;
    int model_embd_dim_ = 0;
};

NB_MODULE(_core, m) {
    m.doc() = "intextus native C++ dense embedding core (GGUF-only)";

    nb::class_<IntextusEncoder>(m, "IntextusEncoder")
        .def(nb::init<const std::string&, const std::string&, bool, int, int, int>(),
            nb::arg("gguf_path"), nb::arg("tokenizer_path"),
            nb::arg("do_lower_case"), nb::arg("num_threads"),
            nb::arg("cls_token_id"), nb::arg("sep_token_id"))
        .def("encode", &IntextusEncoder::encode,
            nb::arg("texts"), nb::arg("max_length"), nb::arg("normalize"));
}