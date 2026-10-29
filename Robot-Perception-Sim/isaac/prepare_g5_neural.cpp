// Prepare the already-present SDK neural depth model in an isolated writable SDK path.
#include <sl/Camera.hpp>
#include <iostream>

int main() {
    const auto model = sl::AI_MODELS::NEURAL_DEPTH;
    const auto before = sl::checkAIModelStatus(model, 0);
    std::cout << "before downloaded=" << before.downloaded << " optimized=" << before.optimized << std::endl;
    if (!before.downloaded) {
        std::cerr << "Required model is absent; refusing an implicit download." << std::endl;
        return 2;
    }
    const auto result = sl::optimizeAIModel(model, 0);
    const auto after = sl::checkAIModelStatus(model, 0);
    std::cout << "optimization=" << result << " after downloaded=" << after.downloaded
              << " optimized=" << after.optimized << std::endl;
    return result == sl::ERROR_CODE::SUCCESS && after.optimized ? 0 : 1;
}
