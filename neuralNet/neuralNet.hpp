#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <memory>
#include <vector>

using namespace std;

class NeuralNet {
private:
    int nLayers;
    int layerPos;
    double** layersInput;
    double** weights;
    double* biases;
    vector<int> network;
    vector<vector<double>> data;

public:
    NeuralNet(vector<vector<double>>& data, vector<int>& network);
    ~NeuralNet();
    void init();
    void initWeight(double* w, int& dim);
    void initNeurons(double* neurons, int& dim);
    void initBiases();
    int getnLayers() { return this->nLayers;};
    
    void train();
    void feedForward(double* input);
    void forward();
    void backpropage();
    double* dotProduct(double* X, double* W, const int nrow, const int ncol);


    double sigmoid(double& val);
    double error(double* t, double* y);
    
    void sumCpu();
    void setBiases(double b[], const int n);
    void setWeights(double* w[]);
    void showW();
    void showN();
};
#endif