// cached_fine_tune_model.h
#pragma once

#include "tensorrt_infer.h" // Include the new header
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory> // For std::unique_ptr

constexpr int VGG16_FEATURE_SIZE = 25088;
constexpr int NUM_CLASSES = 101;

class CachedFineTuneModelImpl : public torch::nn::Module {
public:
    // TensorRT engine for the feature extractor
    std::unique_ptr<TensorRTInfer> feature_extractor;

    // The rest is the same...
    torch::nn::Sequential classifier{nullptr};
    torch::Tensor frozen_data;
    torch::Tensor is_cached_flag;

    CachedFineTuneModelImpl(const std::string& onnx_path, int num_records);
    void cache_activations(
        torch::data::DataLoader<torch::data::datasets::MapDataset<torch::data::datasets::TensorDataset, torch::data::transforms::Stack<torch::data::Example<torch::Tensor, torch::Tensor>>>>& dataloader,
        torch::Device device);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(CachedFineTuneModel);