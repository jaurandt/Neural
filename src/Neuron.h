#pragma once
#include <vector>
#include <cstdlib>

struct Connection
{
    double weight = 0.0;
    double deltaWeight = 0.0;
};

class Neuron;
using Layer = std::vector<Neuron>;

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void feedForward(const Layer& prevLayer);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal() const { return m_outputVal; }
    void calculateOutputGradients(double targetVal);
    void calculateHiddenGradients(const Layer& nextLayer);
    void updateInputWeights(Layer& prevLayer);

private:
    static constexpr double eta = 0.15;   // [0.0, ..., 1.0] overall net learning rate
    static constexpr double alpha = 0.3;  // [0.0, ..., n] momentum multiplier
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight() { return 0.2 * (2.0 * rand() / double(RAND_MAX) - 1.0); }
    double sumDOW(const Layer& nextLayer) const;
    unsigned m_myIndex;
    double m_outputVal = 0.0;
    double m_gradient = 0.0;
    std::vector<Connection> m_outputWeights;
};
