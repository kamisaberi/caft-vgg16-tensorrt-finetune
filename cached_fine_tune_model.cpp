// cached_fine_tune_model.cpp
#include "cached_fine_tune_model.h"
#include <fmt/core.h>
#include <chrono>

// Constructor now takes ONNX path
CachedFineTuneModelImpl::CachedFineTuneModelImpl(
    const std::string& onnx_path,
    int num_records)
{
    // 1. Initialize the TensorRT inference engine
    feature_extractor = std::make_unique<TensorRTInfer>(onnx_path);
    
    // The rest is the same...
    classifier = register_module("classifier", torch::nn::Sequential(/* ... same as before ... */));
    frozen_data = torch::zeros({num_records, VGG16_FEATURE_SIZE});
    register_buffer("frozen_data_buffer", frozen_data);
    is_cached_flag = torch::tensor(false);
    register_buffer("is_cached_buffer", is_cached_flag);
    std::cout << "CachedFineTuneModel initialized with TensorRT Feature Extractor." << std::endl;
}

// Caching method now uses TensorRT
void CachedFineTuneModelImpl::cache_activations(
    torch::data::DataLoader<torch::data::datasets::MapDataset<torch::data::datasets::TensorDataset, torch::data::transforms::Stack<torch::data::Example<torch::Tensor, torch::Tensor>>>>& dataloader,
    torch::Device device)
{
    std::cout << "--- Phase 1: Caching Activations (using TensorRT) ---" << std::endl;
    this->eval();
    torch::NoGradGuard no_grad;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t current_record_idx = 0;

    // Create a temporary GPU tensor to hold the output from TensorRT
    torch::Tensor output_gpu_tensor = torch::zeros({(long)dataloader.batch_size().value(), VGG16_FEATURE_SIZE}, device);

    for (const auto& batch : *dataloader) {
        torch::Tensor data_batch = batch.data.to(device);
        long current_batch_size = data_batch.size(0);

        // Ensure output tensor has correct size for the current batch
        if (current_batch_size != output_gpu_tensor.size(0)) {
            output_gpu_tensor = torch::zeros({current_batch_size, VGG16_FEATURE_SIZE}, device);
        }

        // Perform inference with TensorRT
        feature_extractor->infer(data_batch, output_gpu_tensor);
        
        // Store activations in the main buffer (on CPU first)
        frozen_data.index_put_({torch::indexing::Slice(current_record_idx, current_record_idx + current_batch_size)}, output_gpu_tensor.cpu());
        
        current_record_idx += current_batch_size;
        // ... progress indicator ...
    }
    
    // Move the entire populated buffer to the target device
    frozen_data = frozen_data.to(device);
    is_cached_flag.fill_(true);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> caching_duration = end_time - start_time;
    fmt::print("\nTensorRT Caching complete in {:.2f} seconds.\n", caching_duration.count());
}

// The forward method for accelerated training remains UNCHANGED.
// We only change the forward path for validation/pre-caching.
torch::Tensor CachedFineTuneModelImpl::forward(torch::Tensor x) {
    if (this->is_training() && is_cached_flag.item<bool>()) {
        torch::Tensor cached_batch = frozen_data.index({x});
        return classifier->forward(cached_batch);
    } else {
        // Validation now also uses TensorRT
        torch::NoGradGuard no_grad;
        torch::Tensor output_features = torch::zeros({x.size(0), VGG16_FEATURE_SIZE}, x.device());
        feature_extractor->infer(x, output_features);
        return classifier->forward(output_features);
    }
}