#include <iostream>
#include "neuralNet.hpp"
#include <vector>

using namespace std;
int main(int argc, char* argv[]) {
    vector<double> data{1, 0, 1};
    vector<int> shape{3, 3, 1};
    NeuralNet net = NeuralNet(data, shape);
    cout << net.getnLayers() << endl;

    return 1;
}