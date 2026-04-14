#include "TrainingData.h"
#include <sstream>

TrainingData::TrainingData(const std::string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

void TrainingData::getTopology(std::vector<unsigned>& topology)
{
    std::string line;
    std::string label;

    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    ss >> label;
    if(this->isEOF() || label.compare("topology:") != 0) {
        abort();
    }

    unsigned n = 0;
    while(ss >> n) {
        topology.push_back(n);
    }
}

unsigned TrainingData::getNextInputs(std::vector<double>& inputVals)
{
    inputVals.clear();

    std::string line;
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    std::string label;

    ss >> label;
    if(label.compare("in:") == 0) {
        double oneValue;
        while(ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double>& targetOutputVals)
{
    targetOutputVals.clear();

    std::string line;
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    std::string label;

    ss >> label;
    if(label.compare("out:") == 0) {
        double oneValue;
        while(ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}
