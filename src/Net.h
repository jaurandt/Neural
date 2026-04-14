#pragma once
#include "Neuron.h"
#include <vector>

class Net
{
public:
    Net(const std::vector<unsigned>& topology);
    void feedForward(const std::vector<double>& inputVals);
    void backPropagation(const std::vector<double>& targetVals);
    void getResults(std::vector<double>& resultVals) const;
    double getRecentAverageError() const { return m_recentAverageError; }

private:
    std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_recentAverageError = 0.0;
    static constexpr double m_recentAverageSmoothingFactor = 10.0;
};
