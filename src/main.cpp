#include <iostream>
#include <vector>
#include "TrainingData.h"
#include "Net.h"

static void showVectorVals(const std::string& label, const std::vector<double>& vals)
{
    std::cout << label;
    for(unsigned i = 0; i < vals.size(); ++i) {
        std::cout << vals[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    const std::string dataFile = (argc > 1) ? argv[1] : "data/trainingData.txt";
    TrainingData trainingData(dataFile);

    std::vector<unsigned> topology;
    trainingData.getTopology(topology);

    Net myNet(topology);

    std::vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while(trainingData.getNextInputs(inputVals))
    {
        ++trainingPass;
        std::cout << std::endl << "Pass #" << trainingPass << std::endl;

        showVectorVals("Inputs: ", inputVals);
        myNet.feedForward(inputVals);

        myNet.getResults(resultVals);
        showVectorVals("Outputs: ", resultVals);

        trainingData.getTargetOutputs(targetVals);
        showVectorVals("Targets: ", targetVals);

        myNet.backPropagation(targetVals);

        std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;
    }

    std::cin.get();

    return 0;
}
