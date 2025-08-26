// tensorrt_infer.cpp
#include "tensorrt_infer.h"
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <fmt/core.h>

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

// Helper to check for CUDA errors
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fmt::print("CUDA Error: {} at {} : {}\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

TensorRTInfer::TensorRTInfer(const std::string& onnx_path) {
    buildEngineFromONNX(onnx_path);

    if (!engine_) {
        std::cerr << "Failed to create TensorRT engine." << std::endl;
        exit(EXIT_FAILURE);
    }
    context_ = engine_->createExecutionContext();
    
    // Allocate GPU buffers
    // Assuming one input and one output
    input_size_ = engine_->getBindingBytesCount(0);
    output_size_ = engine_->getBindingBytesCount(1);
    
    CHECK_CUDA(cudaMalloc(&input_buffer_, input_size_));
    CHECK_CUDA(cudaMalloc(&output_buffer_, output_size_));
}

TensorRTInfer::~TensorRTInfer() {
    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (input_buffer_) CHECK_CUDA(cudaFree(input_buffer_));
    if (output_buffer_) CHECK_CUDA(cudaFree(output_buffer_));
}

void TensorRTInfer::buildEngineFromONNX(const std::string& onnx_path) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger_);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger_);
    
    std::ifstream onnx_file(onnx_path, std::ios::binary);
    if (!onnx_file.good()) {
        std::cerr << "Could not read ONNX file: " << onnx_path << std::endl;
        return;
    }
    
    onnx_file.seekg(0, onnx_file.end);
    size_t size = onnx_file.tellg();
    onnx_file.seekg(0, onnx_file.beg);
    
    std::vector<char> onnx_model(size);
    onnx_file.read(onnx_model.data(), size);
    
    if (!parser->parse(onnx_model.data(), size)) {
        std::cerr << "Failed to parse the ONNX file." << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // Allow TensorRT to use up to 1GB of GPU memory for tactics
    config->setMaxWorkspaceSize(1 << 30); 
    
    // If your GPU supports FP16, you can enable it for faster inference
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    engine_ = builder->buildEngineWithConfig(*network, *config);
    
    // Cleanup
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}

void TensorRTInfer::infer(torch::Tensor input, torch::Tensor& output) {
    if (input.device().type() != torch::kCUDA) {
        std::cerr << "Input tensor must be on a CUDA device." << std::endl;
        return;
    }

    // Check tensor is contiguous
    input = input.contiguous();

    // Copy input data from PyTorch tensor to GPU buffer
    CHECK_CUDA(cudaMemcpy(input_buffer_, input.data_ptr(), input.nbytes(), cudaMemcpyDeviceToDevice));
    
    // Execute inference
    void* bindings[] = {input_buffer_, output_buffer_};
    context_->executeV2(bindings);

    // Copy output data from GPU buffer back to PyTorch tensor
    CHECK_CUDA(cudaMemcpy(output.data_ptr(), output_buffer_, output.nbytes(), cudaMemcpyDeviceToDevice));
}