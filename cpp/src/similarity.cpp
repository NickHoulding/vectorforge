#include "vectorforge/similarity.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

py::array_t<float> cosine_similarity_batch(
    py::array_t<float> query_embedding,
    py::array_t<float> doc_embeddings
) {
    py::buffer_info query_buf = query_embedding.request();
    py::buffer_info docs_buf = doc_embeddings.request();

    if (query_buf.ndim != 1 || docs_buf.ndim != 2) {
        throw std::runtime_error("Invalid array dimensions");
    }

    size_t embed_dim = query_buf.shape[0];
    size_t n_docs = docs_buf.shape[0];
    size_t doc_dim = docs_buf.shape[1];

    if (embed_dim != doc_dim) {
        throw std::runtime_error("Dimension mismatch between query and documents");
    }

    float* query_ptr = static_cast<float*>(query_buf.ptr);
    float* docs_ptr = static_cast<float*>(docs_buf.ptr);

    auto result = py::array_t<float>(n_docs);
    py::buffer_info result_buf = result.request();
    float* result_ptr = static_cast<float*>(result_buf.ptr);

    for (size_t i = 0; i < n_docs; i++) {
        float dot_product = 0.0f;

        for (size_t j = 0; j < embed_dim; j++) {
            dot_product += query_ptr[j] * docs_ptr[i * embed_dim + j];
        }

        result_ptr[i] = dot_product;
    }

    return result;
}
