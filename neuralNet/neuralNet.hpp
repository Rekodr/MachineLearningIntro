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
    unsigned sampleInputIdx;
    unsigned epochs;
    unsigned miniBatchSize;
    double learningRate;

    Layers layersInput;
    Layers layersWeights;
    Layers layersError;
    vector<unsigned> network;
    vector<vector<double>> data;
    vector<vector<double>> targets;

public:
    NeuralNet(vector<vector<double>>& input, vector<vector<double>>& targets,  vector<unsigned>& network,
        double learningRate=0.5, unsigned epochs=1, unsigned miniBatchSize=1);
    ~NeuralNet();
    void init();
    void initWeight(Layer w, unsigned& dim, unsigned& nNodes);
    void initNeurons(Layer neurons, unsigned& dim);
    void initBiases();
    unsigned getnLayers() { return this->nLayers;};
    
    void train();
    void test(vector<vector<double>>& input, vector<vector<double>>& targets);
    unsigned classify(vector<double>& input);
    double feedForward();
    void forward();
    void backPropagation();
    void backPropageError(unsigned layerIndex, double* target=nullptr);
    void learn();
    
    Layer dotProduct(Layer X, Layer W, const unsigned nrow, const unsigned ncol, string transfer="");
    void fetchInput();
    
    double sigmoid(double& val);
    double totalError(Layer target);
    void clearErrors();
    void avgErrors(Layer deltas, unsigned& dim, unsigned N);
    
    void setBiases(double b[], const unsigned n);
    void setWeights(Layer w[]);
    void showW();
    void showN();
};
#endif
