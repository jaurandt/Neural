#include "Neuron.h"
#include <cmath>

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) : m_myIndex(myIndex)
{
    for(unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for(unsigned n = 0; n < prevLayer.size() - 1; ++n) {
        Neuron& neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
            // Individual input, magnified by the gradient and train rate
            eta
            * neuron.getOutputVal()
            * m_gradient
            // Add momentum: a fraction of the previous delta weight
            + alpha
            * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
    double sum = 0.0;

    // Sum contributions of the errors at the nodes we feed
    for(unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calculateHiddenGradients(const Layer& nextLayer)
{
    double dow = Neuron::sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calculateOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return (1 - (x * x)); // approximation of tanh derivative
}

void Neuron::feedForward(const Layer& prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (our inputs) including the bias node
    for(unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
            prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}
