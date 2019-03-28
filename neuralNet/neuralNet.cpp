#include <iostream>
#include "neuralNet.hpp"

using namespace std;
NeuralNet::NeuralNet(vector<double> data, vector<int>& network) {
    if(network.size() < 3) {
        throw "In valid shape.";
    }

    this->network.assign(network.begin(), network.end()); 
    this->nLayers = network.size();
    this->init();
}

NeuralNet::~NeuralNet() {
    for(auto i = 0; i < this->nLayers; i++) {
        delete this->hlayers[i];
    }

    cout << sizeof(this->weights) << endl;
    for(auto i = 0; i < this->nLayers - 1; i++) {
        delete this->weights[i];
    }
    
    delete this->hlayers;
    delete this->biases;
    delete this->weights;
}

void NeuralNet::init() {
    this->hlayers = new double*[this->nLayers];
    this->biases = new double[this->nLayers - 1]; 
    this->weights = new double*[this->nLayers - 1];

    this->initBiases(this->biases);
    
    for(auto i = 0; i < this->nLayers; i++) {
        int layerDim = this->network.at(i);
        this->hlayers[i] = new double[layerDim];
        this->initNeurons(this->hlayers[i], layerDim);
        if(i > 0) {
            int prevLayerDim = this->network.at(i-1);
            int wDim = prevLayerDim * layerDim;
            this->weights[i-1] = new double[wDim];
            this->initWeight(this->weights[i-1], wDim);
        }
    }
    cout << "Hello" << endl;
}

void NeuralNet::initBiases(double *b) {
    for(auto i = 0; i < (this->nLayers - 1); i++) {
        b[i] = 0.0; // need to randomize
    }
}

void NeuralNet::initNeurons(double* neurons, int& dim) {
    for(int i = 0; i < dim; i++) {
        neurons[i] = 0.0;  // need to randomize
    }
}

void NeuralNet::initWeight(double* w, int& dim) {
    for(auto i = 0; i < dim; i++) {
        w[i] = 0.0; // need to randomize
    }
}