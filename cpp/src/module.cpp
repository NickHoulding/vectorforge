#include <pybind11/pybind11.h>
#include "vectorforge/similarity.h"

PYBIND11_MODULE(vectorforge_cpp, m) {
    m.doc() = "VectorForge C++ extension module for high-performance vector operations";

    m.def("cosine_similarity_batch", &cosine_similarity_batch,
          "Compute cosine similarity between a query and multiple documents",
          pybind11::arg("query_embedding"),
          pybind11::arg("doc_embeddings"));
}
