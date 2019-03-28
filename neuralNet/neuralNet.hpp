#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <memory>
#include <vector>

using namespace std;

class NeuralNet {
private:
    int nLayers;
    double** layersInput;
    double** weights;
    double* biases;
    vector<int> network;
public:
    NeuralNet(vector<double> data, vector<int>& network);
    ~NeuralNet();
    void init();
    void initWeight(double* w, int& dim);
    void initNeurons(double* neurons, int& dim);
    void initBiases(double* b);
    int getnLayers() { return this->nLayers;};

    void showW();
    void showN();
    void showB();
};
#endif