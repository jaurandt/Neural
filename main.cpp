#include <iostream>
#include <fstream>
#include <sstream> 
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>

using namespace std;

class TrainingData
{
    public:
        TrainingData(const string filename);
        bool isEOF() { return m_trainingDataFile.eof(); }
        void getTopology(vector<unsigned>& topology);
        unsigned getNextInputs(vector<double>& inputVals);
        unsigned getTargetOutputs(vector<double>& targetOutputVals);
    
    private:
        ifstream m_trainingDataFile;

};

void TrainingData::getTopology(vector<unsigned>& topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if(this->isEOF() || label.compare("topology:") != 0) {
        abort();
    }

    unsigned n = 0;
    while(ss >> n) {
        topology.push_back(n);
    }
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double>& inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    string label;

    ss >> label;
    if(label.compare("in:") == 0) {
        double oneValue;
        while(ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double>& targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    string label;

    ss >> label;
    if(label.compare("out:") == 0) {
        double oneValue;
        while(ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

//typedef vector<Neuron> Layer;
using Layer = vector<Neuron>;

class Neuron
{
    public:
        Neuron(unsigned numOutputs, unsigned myIndex, const double eta, const double alpha);
        void feedForward(const Layer& prevLayer);
        void setOutputVal(double val) { m_outputVal = val; }
        double getOutputVal() const { return m_outputVal; }
        void calculateOutputGradients(double targetVal);
        void calculateHiddenGradients(const Layer& nextLayer);
        void updateInputWeights(Layer& prevLayer);
        
    private:
        static double eta;    // [0.0, ..., 1.0] overall net training rate
        static double alpha; // [0.0, ..., n] multiplier of last weight change (momentum)
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        static double randomWeight() { return rand() / double(RAND_MAX); }
        double sumDOW(const Layer& nextLayer) const;
        unsigned m_myIndex;
        double m_outputVal;
        double m_gradient;
        vector<Connection> m_outputWeights;
};

void Neuron::updateInputWeights(Layer& prevLayer)
{
    //The weights to be updated are in the Connection container
    //in the neurons in the preceding layer

    for(unsigned n = 0; n < prevLayer.size() - 1; ++n) {
        Neuron& neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
            //Individual input, magnified by the gradient and train rate
            eta //overall learning rate [0.0, ..., 1.0]
            * neuron.getOutputVal()
            * m_gradient
            //Also add momentum = a fraction of the previous delta weight
            + alpha //momentum
            * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
    double sum = 0.0;

    ////////Left off here at 53:30 / 1:05:24////////////

    //Sum contributions of the errors at the nodes we feed

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
    return (1 - (x * x)); //approximation of tanh derivative
}

void Neuron::feedForward(const Layer& prevLayer)
{
    double sum = 0.0;

    //sum the previous layer's outputs (our inputs) including bias node
    for(unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
            prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex, const double eta, const double alpha) : m_myIndex(myIndex)
{
    Neuron::eta = eta;
    Neuron::alpha = alpha;
    for(unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
}

// Initialize static members
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

class Net
{
    public:
        Net(const vector<unsigned>& topology);
        void feedForward(const vector<double>& inputVals);
        void backPropagation(const vector<double>& targetVals);
        void getResults(vector<double>& resultVals) const;
        double getRecentAverageError() const { return m_recentAverageError; }

    private:
        vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
        double m_error;
        double m_recentAverageError = 0.0;
        double m_recentAverageSmoothingFactor = 10.0;
};

void Net::getResults(vector<double>& resultVals) const
{
    resultVals.clear();

    const Layer& outputLayer = m_layers.back();
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        resultVals.push_back(outputLayer[n].getOutputVal());
    }
}

void Net::backPropagation(const vector<double>& targetVals) 
{
    //Calculate overall net error
    //RMS of output neuron errors
    double delta;
    Layer& outputLayer = m_layers.back();
    m_error = 0.0;

    //Calculating RMS error
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }

    m_error /= outputLayer.size() - 1; //average error squared (-1 for bias neuron)
    m_error = sqrt(m_error); //RMS

    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
                            / (m_recentAverageSmoothingFactor + 1.0);

    //Calculate output layer gradients

    for(unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calculateOutputGradients(targetVals[n]);
    }

    //Calculate gradients on hidden layers

    for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer& hiddenLayer = m_layers[layerNum]; //optimize
        Layer& nextLayer = m_layers[layerNum + 1]; //optimize

        for(unsigned n = 0; n < hiddenLayer.size() - 1; ++n) {
            hiddenLayer[n].calculateHiddenGradients(nextLayer);
        }
    }

    //For all layers from outputs to first hidden layer, update connection weights

    for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer& layer = m_layers[layerNum];
        Layer& prevLayer = m_layers[layerNum - 1];

        for(unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<double>& inputVals)
{
    assert(inputVals.size() == (m_layers[0].size() - 1)); // -1 for bias neuron

    for(unsigned i = 0; i < inputVals.size(); ++i) {
        //setting the value of the input neuron layer
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    //forward propagation
    for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer& prevLayer = m_layers[layerNum - 1]; //optimize
        for(unsigned n = 0; n < m_layers[layerNum].size(); ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned>& topology)
{
    unsigned numLayers = topology.size();

    unsigned numOutputs;

    //overall Net learning rate and momentum
    const double eta = 0.1; //learning rate
    const double alpha = 0.5; //momentum

    //net has layers
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        numOutputs = (layerNum == (topology.size() - 1)) ? 0 : topology[layerNum + 1];
        //layer has neurons
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum, eta, alpha));
            cout << "Neuron " << layerNum << ", " << neuronNum << " created!" << endl;
        }

        //force the bias node's output value to 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}

int main(int argc, char** argv) 
{
    TrainingData trainingData("trainingData.txt");
    
    // such as {3, 2, 1} (layers)
    vector<unsigned> topology;

    trainingData.getTopology(topology);
    //topology.push_back(3);
    //topology.push_back(2);
    //topology.push_back(1);

    Net myNet(topology);

    //vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    vector<double> inputVals;
        //myNet.feedForward(inputVals);

    vector<double> targetVals;
        //myNet.backPropagation(targetVals);

    vector<double> resultVals;
        //myNet.getResults(resultVals);

    
    while(trainingData.getNextInputs(inputVals))
    {
        ++trainingPass;
        cout << endl << "Pass #" << trainingPass;

        //Get new input data and feed it forward
        
        
        //showVectorVals()
        cout << endl << "Inputs: ";
        for(unsigned i = 0; i < inputVals.size(); ++i)
        {
            cout << inputVals[i] << " ";
        }
        
        myNet.feedForward(inputVals);

        //Collect the net's actual results
        myNet.getResults(resultVals);
        
        //showVectorVals()
        cout << endl << "Outputs: ";
        for(unsigned i = 0; i < resultVals.size(); ++i)
        {
            cout << resultVals[i] << " ";
        }

        //Train the net what the outputs should have been
        trainingData.getTargetOutputs(targetVals);
        
        //showVectorVals()
        cout << endl << "Targets: ";
        for(unsigned i = 0; i < targetVals.size(); ++i)
        {
            cout << targetVals[i] << " ";
        }

        //cout << "targetVals.size() -> " << targetVals.size() << endl;
        //cout << "topology.back() -> " << topology.back() << endl;
        //assert(targetVals.size() == topology.back());

        myNet.backPropagation(targetVals);

        cout << endl << "Net recent average error: "
             << myNet.getRecentAverageError() << endl;
    }

    return 0;
}