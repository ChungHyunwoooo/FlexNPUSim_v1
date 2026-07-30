/**
 * @file function_model_if.h
 * @brief DNN Operation Functional Model Interfaces
 *
 * Abstract interfaces for DNN computational operations.
 * Used for functional verification (not timing-accurate).
 * All operations are pure C++ (no SystemC dependency).
 */

#pragma once

#include "compiler/dnn_image/dnn_image_format.h"
#include <vector>
#include <string>
#include <cstdint>

namespace flexnpu_sim {

// ============================================================================
// Tensor Representation
// ============================================================================

/**
 * @brief Tensor data structure for function models
 *
 * Represents a 4D tensor (batch, channels, height, width).
 * Data is stored in NCHW layout (batch-major, channel-major).
 */
struct Tensor {
    std::vector<float> data;   ///< Flattened tensor data (NCHW layout)
    uint32_t height = 0;       ///< Spatial height
    uint32_t width = 0;        ///< Spatial width
    uint32_t channels = 0;     ///< Number of channels
    uint32_t batch = 1;        ///< Batch size

    /**
     * @brief Total number of elements
     */
    size_t size() const {
        return static_cast<size_t>(height) * width * channels * batch;
    }

    /**
     * @brief Access element (mutable)
     * @param h Height index
     * @param w Width index
     * @param c Channel index
     * @param b Batch index
     */
    float& at(uint32_t h, uint32_t w, uint32_t c, uint32_t b = 0);

    /**
     * @brief Access element (const)
     */
    const float& at(uint32_t h, uint32_t w, uint32_t c, uint32_t b = 0) const;
};

// ============================================================================
// Pooling Types
// ============================================================================

/**
 * @brief Pooling operation type
 */
enum class PoolingType : uint32_t {
    Max = 0,      ///< Max pooling
    Average = 1   ///< Average pooling
};

// ============================================================================
// Activation Types
// ============================================================================

/**
 * @brief Activation function type
 */
enum class ActivationType : uint32_t {
    ReLU = 0,     ///< Rectified Linear Unit
    ReLU6 = 1,    ///< ReLU clamped to [0, 6]
    Sigmoid = 2,  ///< Sigmoid activation
    Tanh = 3      ///< Hyperbolic tangent
};

// ============================================================================
// Abstract Function Model Interface
// ============================================================================

/**
 * @brief Abstract interface for DNN operation functional models
 *
 * All DNN operations (Conv2D, Pooling, etc.) implement this interface.
 * Models perform functional computation without timing simulation.
 */
class FunctionModelIf {
public:
    virtual ~FunctionModelIf() = default;

    /**
     * @brief Execute forward pass
     * @param input Input tensor
     * @param weight Weight tensor (empty for operations without weights)
     * @return Output tensor
     */
    virtual Tensor forward(const Tensor& input, const Tensor& weight) = 0;

    /**
     * @brief Human-readable operation name
     */
    virtual std::string name() const = 0;

    /**
     * @brief Layer type identifier
     */
    virtual FuncType layer_type() const = 0;
};

// ============================================================================
// Conv2D Model
// ============================================================================

/**
 * @brief Configuration for 2D convolution
 */
struct Conv2dConfig {
    uint32_t stride = 1;       ///< Convolution stride
    uint32_t padding = 0;      ///< Zero padding
    uint32_t dilation = 1;     ///< Dilation factor
    uint32_t groups = 1;       ///< Number of groups (1 = standard conv)
};

/**
 * @brief 2D Convolution functional model
 */
class Conv2dModel : public FunctionModelIf {
public:
    explicit Conv2dModel(const Conv2dConfig& config);

    Tensor forward(const Tensor& input, const Tensor& weight) override;
    std::string name() const override;
    FuncType layer_type() const override;

private:
    Conv2dConfig config_;
};

// ============================================================================
// Depthwise Conv2D Model
// ============================================================================

/**
 * @brief Configuration for depthwise 2D convolution
 */
struct DepthwiseConv2dConfig {
    uint32_t stride = 1;       ///< Convolution stride
    uint32_t padding = 0;      ///< Zero padding
    uint32_t dilation = 1;     ///< Dilation factor
};

/**
 * @brief Depthwise 2D Convolution functional model
 */
class DepthwiseConv2dModel : public FunctionModelIf {
public:
    explicit DepthwiseConv2dModel(const DepthwiseConv2dConfig& config);

    Tensor forward(const Tensor& input, const Tensor& weight) override;
    std::string name() const override;
    FuncType layer_type() const override;

private:
    DepthwiseConv2dConfig config_;
};

// ============================================================================
// Pooling Model
// ============================================================================

/**
 * @brief Configuration for pooling operation
 */
struct PoolingConfig {
    PoolingType type = PoolingType::Max;  ///< Pooling type
    uint32_t kernel_size = 2;              ///< Pooling window size
    uint32_t stride = 2;                   ///< Pooling stride
    uint32_t padding = 0;                  ///< Zero padding
};

/**
 * @brief Pooling functional model
 */
class PoolingModel : public FunctionModelIf {
public:
    explicit PoolingModel(const PoolingConfig& config);

    Tensor forward(const Tensor& input, const Tensor& weight) override;
    std::string name() const override;
    FuncType layer_type() const override;

private:
    PoolingConfig config_;
};

// ============================================================================
// Fully Connected Model
// ============================================================================

/**
 * @brief Configuration for fully connected layer
 */
struct FullyConnectedConfig {
    uint32_t input_features = 0;   ///< Number of input features
    uint32_t output_features = 0;  ///< Number of output features
    bool use_bias = true;          ///< Whether to use bias
};

/**
 * @brief Fully Connected (Dense) functional model
 */
class FullyConnectedModel : public FunctionModelIf {
public:
    explicit FullyConnectedModel(const FullyConnectedConfig& config);

    Tensor forward(const Tensor& input, const Tensor& weight) override;
    std::string name() const override;
    FuncType layer_type() const override;

private:
    FullyConnectedConfig config_;
};

// ============================================================================
// Activation Model
// ============================================================================

/**
 * @brief Configuration for activation function
 */
struct ActivationConfig {
    ActivationType type = ActivationType::ReLU;  ///< Activation type
};

/**
 * @brief Activation function functional model
 */
class ActivationModel : public FunctionModelIf {
public:
    explicit ActivationModel(const ActivationConfig& config);

    Tensor forward(const Tensor& input, const Tensor& weight) override;
    std::string name() const override;
    FuncType layer_type() const override;

private:
    ActivationConfig config_;
};

// ============================================================================
// MatMul Model
// ============================================================================

/**
 * @brief Matrix multiplication mode for attention mechanism
 */
enum class MatMulMode : uint32_t {
    QKt = 0,    ///< Q × K^T (attention scores)
    AV  = 1     ///< A × V (attention output)
};

/**
 * @brief Configuration for matrix multiplication
 */
struct MatMulConfig {
    MatMulMode mode = MatMulMode::QKt;
};

/**
 * @brief Matrix Multiplication functional model
 *
 * Supports attention mechanism matrix multiplications:
 *   QKt: input[b,h] × weight[b,h]^T → [seq_len, seq_len]
 *   AV:  input[b,h] × weight[b,h]   → [seq_len, d_k]
 */
class MatMulModel : public FunctionModelIf {
public:
    explicit MatMulModel(const MatMulConfig& config);

    Tensor forward(const Tensor& input, const Tensor& weight) override;
    std::string name() const override;
    FuncType layer_type() const override;

private:
    MatMulConfig config_;
};

// ============================================================================
// Element-wise Model
// ============================================================================

/**
 * @brief Element-wise operation type
 */
enum class ElementWiseOp : uint32_t {
    Add      = 0,  ///< output = input + weight
    Multiply = 1,  ///< output = input * weight
    Subtract = 2   ///< output = input - weight
};

/**
 * @brief Configuration for element-wise operation
 */
struct ElementWiseConfig {
    ElementWiseOp op = ElementWiseOp::Add;
};

/**
 * @brief Element-wise functional model
 *
 * Applies Add, Multiply, or Subtract element-wise between two same-shape tensors.
 */
class ElementWiseModel : public FunctionModelIf {
public:
    explicit ElementWiseModel(const ElementWiseConfig& config);

    Tensor forward(const Tensor& input, const Tensor& weight) override;
    std::string name() const override;
    FuncType layer_type() const override;

private:
    ElementWiseConfig config_;
};

} // namespace flexnpu_sim
