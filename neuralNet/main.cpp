#include <iostream>
#include "neuralNet.hpp"

using namespace std;
int main(int argc, char* argv[]) {
    NeuralNet net = NeuralNet();
    cout << net.getnLayers() << endl;

    return 1;
}