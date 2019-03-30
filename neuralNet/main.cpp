#include <iostream>
#include "neuralNet.hpp"
#include <vector>

using namespace std;
int main(int argc, char* argv[]) {
    vector<unsigned> shape{2, 2, 1};

    vector<vector<double>> data = {{0, 1}};
    vector<vector<double>> targets = {{1}};
    double b[] = {1, 1} ;

    double tmp[2][2][3] = {
        { {1, 0.5, 1}, {-1, 2, 1} },
        { {1.5, -1, 1}}
    }; 

    double** w = new double*[2];
    for(auto i = 0; i < 2; i++) {
        int prev = shape.at(i) + 1;
        int curr = shape.at(i+1);
        w[i] = new double[prev * curr];
        memcpy(w[i], tmp[i], sizeof(double) * curr * prev);
    }

    NeuralNet net = NeuralNet(data, targets, shape);
    net.setBiases(b, 2);
    net.setWeights((double**)w);

    cout << net.getnLayers() << endl << endl;
    cout << "W" << endl;
    net.showW();
    cout << endl;
    cout << endl;
    cout << "X" << endl;
    net.showN();

    vector<double> x = data.at(0);
    net.feedForward(x.data());

    return 1;
}