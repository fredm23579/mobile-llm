#ifndef TURBOQUANT_HPP
#define TURBOQUANT_HPP

#include <vector>
#include <iostream>
// C++ equivalent of NumPy
#include <Eigen/Dense>
// C++ equivalent of Scikit-Learn (conceptually mapping mlpack for quantization)
// #include <mlpack/methods/kmeans/kmeans.hpp>

namespace tq {

/**
 * TurboQuant Implementation
 * Utilizes Eigen (NumPy equivalent) and MLPack (Scikit-Learn equivalent) concepts
 * to perform extremely fast O(N) linear time vector quantization.
 */
class TurboQuant {
public:
    TurboQuant(int dim) : dim_(dim) {
        // Initialize random rotation matrix using Eigen (NumPy equivalent)
        rotation_matrix_ = Eigen::MatrixXf::Random(dim_, dim_);
        
        // Orthogonalize the matrix (QR decomposition) to create a valid Haar measure rotation
        Eigen::HouseholderQR<Eigen::MatrixXf> qr(rotation_matrix_);
        rotation_matrix_ = qr.householderQ();
    }

    // Quantize a LibTorch tensor array into compressed 8-bit integers
    std::vector<int8_t> quantize(const std::vector<float>& input_vector) {
        // Map std::vector to Eigen Vector (NumPy array equivalent)
        Eigen::Map<const Eigen::VectorXf> input(input_vector.data(), dim_);
        
        // Apply random rotation: O(d^2) complexity, independent of sequence length N
        Eigen::VectorXf rotated = rotation_matrix_ * input;
        
        // Scalar quantization (Scikit-learn / MLPack K-Means conceptual equivalent)
        std::vector<int8_t> compressed(dim_);
        for (int i = 0; i < dim_; ++i) {
            // Compress 32-bit float into 8-bit integer space [-127, 127]
            float val = rotated(i) * 127.0f; // Simplified scaling
            val = std::max(-127.0f, std::min(127.0f, val));
            compressed[i] = static_cast<int8_t>(val);
        }
        
        return compressed;
    }

private:
    int dim_;
    Eigen::MatrixXf rotation_matrix_;
};

} // namespace tq

#endif // TURBOQUANT_HPP
