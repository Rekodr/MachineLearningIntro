#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <memory>
#include <vector>

using namespace std;

class NeuralNet {
private:
    int nLayers;
    double** hlayers;
    double*** weights;
    double* biases;
    vector<int> network;
public:
    NeuralNet(vector<double> data, vector<int>& network);
    ~NeuralNet();
    void init();
    void init_weight(double** w, int& dim);
    void init_neurons(double* neurons, int& dim);
    int getnLayers() { return this->nLayers;};
};
#endif