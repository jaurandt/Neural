#include "Net.h"
#include <iostream>
#include <cassert>
#include <cmath>

Net::Net(const std::vector<unsigned>& topology)
{
    unsigned numLayers = topology.size();

    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = (layerNum == (topology.size() - 1)) ? 0 : topology[layerNum + 1];

        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Neuron " << layerNum << ", " << neuronNum << " created!" << std::endl;
        }

        m_layers.back().back().setOutputVal(1.0); // bias node
    }
}

void Net::getResults(std::vector<double>& resultVals) const
{
    resultVals.clear();

    const Layer& outputLayer = m_layers.back();
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        resultVals.push_back(outputLayer[n].getOutputVal());
    }
}

void Net::backPropagation(const std::vector<double>& targetVals)
{
    Layer& outputLayer = m_layers.back();

    // Calculate RMS error across output neurons
    double error = 0.0;
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        error += delta * delta;
    }
    error /= outputLayer.size() - 1; // average error squared (-1 for bias neuron)
    error = sqrt(error);             // RMS

    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + error)
                            / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calculateOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers (back to front)
    for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer& hiddenLayer = m_layers[layerNum];
        Layer& nextLayer = m_layers[layerNum + 1];

        for(unsigned n = 0; n < hiddenLayer.size() - 1; ++n) {
            hiddenLayer[n].calculateHiddenGradients(nextLayer);
        }
    }

    // Update connection weights from outputs back to first hidden layer
    for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer& layer = m_layers[layerNum];
        Layer& prevLayer = m_layers[layerNum - 1];

        for(unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const std::vector<double>& inputVals)
{
    assert(inputVals.size() == (m_layers[0].size() - 1)); // -1 for bias neuron

    for(unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagation
    for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer& prevLayer = m_layers[layerNum - 1];

        for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }

        m_layers[layerNum].back().setOutputVal(1.0); // bias node
    }
}
