#include <iostream>
#include "neuralNet.hpp"
#include <vector>

using namespace std;
int main(int argc, char* argv[]) {
    vector<double> data{1, 0, 1};
    vector<int> shape{2, 2, 1};
    NeuralNet net = NeuralNet(data, shape);
    cout << net.getnLayers() << endl << endl;
    cout << "W" << endl;
    net.showW();
    cout << endl;
    cout << "B" << endl;
    net.showB();
    cout << endl;
    cout << "X" << endl;
    net.showN();
    return 1;
}