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

    delete this->hlayers;
}

void NeuralNet::init() {
    this->hlayers = new double*[this->nLayers];
    this->biases = new double[this->nLayers - 1]; 
    this->weights = new double**[this->nLayers - 1];

    for(auto i = 0; i < this->nLayers; i++) {
        int layerDim = this->network.at(i);
        this->hlayers[i] = new double[layerDim];
        this->init_neurons(this->hlayers[i], layerDim);
        if(i > 0) {
            this->weights[i-1] = new double*[layerDim];
            this->init_weight(this->weights[i-1], this->network.at(i-1));
        }
    }
    cout << "Hello" << endl;
}

void NeuralNet::init_neurons(double* neurons, int& dim) {
    for(int i = 0; i < dim; i++) {
        neurons[i] = 0.0;  // need to randomize
    }
}

void NeuralNet::init_weight(double** w, int& dim) {
    for(auto i = 0; i < dim; i++) {
        w[i] = new double[dim];
        for(auto j = 0; j < dim; j++) {
            w[i][0] = 0.0; // need to randomize
        }
    }
}