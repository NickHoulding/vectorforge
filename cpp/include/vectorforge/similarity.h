#ifndef VECTORFORGE_SIMILARITY_H
#define VECTORFORGE_SIMILARITY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * Compute cosine similarity between a query embedding and multiple document embeddings.
 * 
 * This function computes the dot product between the query vector and each document vector.
 * Since embeddings are pre-normalized, the dot product equals the cosine similarity.
 * 
 * @param query_embedding 1D NumPy array of shape (embed_dim,) - the query vector
 * @param doc_embeddings 2D NumPy array of shape (n_docs, embed_dim) - document vectors
 * @return 1D NumPy array of shape (n_docs,) containing similarity scores
 * 
 * @throws std::runtime_error if array dimensions are invalid or mismatched
 */
py::array_t<float> cosine_similarity_batch(
    py::array_t<float> query_embedding,
    py::array_t<float> doc_embeddings
);

#endif
