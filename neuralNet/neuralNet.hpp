#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <memory>
#include <vector>

using namespace std;
typedef double** Layers;
typedef double* Layer;


class NeuralNet {
private:
    unsigned nLayers;
    unsigned layerPos;
    Layers layersInput;
    Layers layersWeights;
    Layers layersError;
    vector<unsigned> network;
    vector<vector<double>> data;
    vector<vector<double>> targets;

public:
    NeuralNet(vector<vector<double>>& input, vector<vector<double>>& targets,  vector<unsigned>& network);
    ~NeuralNet();
    void init();
    void initWeight(double* w, unsigned& dim);
    void initNeurons(double* neurons, unsigned& dim);
    void initBiases();
    unsigned getnLayers() { return this->nLayers;};
    
    void train();
    void feedForward(double* input);
    void forward();
    void backpropagation();
    void backpropage();
    
    Layer dotProduct(Layer X, Layer W, const unsigned nrow, const unsigned ncol);


    double sigmoid(double& val);
    double totalError(double* target);
    double layerError();
    
    void sumCpu();
    void setBiases(double b[], const unsigned n);
    void setWeights(double* w[]);
    void showW();
    void showN();
};
#endif