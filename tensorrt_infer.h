// tensorrt_infer.h
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <torch/torch.h>

// TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
};

class TensorRTInfer {
public:
    TensorRTInfer(const std::string& onnx_path);
    ~TensorRTInfer();

    // Perform inference
    void infer(torch::Tensor input, torch::Tensor& output);

private:
    void buildEngineFromONNX(const std::string& onnx_path);

    Logger logger_;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // GPU buffers
    void* input_buffer_ = nullptr;
    void* output_buffer_ = nullptr;

    // Buffer sizes
    size_t input_size_ = 0;
    size_t output_size_ = 0;
};