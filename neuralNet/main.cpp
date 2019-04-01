#include <iostream>
#include "neuralNet.hpp"
#include "dataReader.hpp"
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

    NeuralNet net = NeuralNet(data, targets, shape, 0.5, 1);
    net.setBiases(b, 2);
    net.setWeights((double**)w);

    cout << "W" << endl;
    net.showW();
    cout << endl;

    net.train();
    net.showW();

    // DataReader reader1 = DataReader("handWrittenDigitNormalized-training.data", make_pair(5, 64), 10);
    // DataReader reader2 = DataReader("handWrittenDigitNormalized-testing.data", make_pair(1797, 64), 10);
    // auto X_train = reader1.getX();
    // auto X_test = reader2.getX();
    // cout << X_train.size() << endl;
    // cout << X_test.size() << endl;

    // vector<unsigned> networkStruct{64, 20, 10};
    // auto X = reader1.getX();
    // auto Y = reader1.getY();
    // NeuralNet neuralNet = NeuralNet(X, Y, networkStruct, 0.5, 1);
    // neuralNet.train();
    
    
    return 1;
}