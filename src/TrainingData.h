#pragma once
#include <string>
#include <vector>
#include <fstream>

class TrainingData
{
public:
    TrainingData(const std::string filename);
    bool isEOF() { return m_trainingDataFile.eof(); }
    void getTopology(std::vector<unsigned>& topology);
    unsigned getNextInputs(std::vector<double>& inputVals);
    unsigned getTargetOutputs(std::vector<double>& targetOutputVals);

private:
    std::ifstream m_trainingDataFile;
};
