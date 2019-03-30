#include <iostream>
#include "neuralNet.hpp"
#include <cmath>

using namespace std;
NeuralNet::NeuralNet(vector<vector<double>>& data, vector<int>& network) {
    if(network.size() < 3) {
        throw "In valid shape.";
    }

    this->data = data;
    this->network.assign(network.begin(), network.end()); 
    this->nLayers = network.size();
    this->init();
}

NeuralNet::~NeuralNet() {
    for(auto i = 0; i < this->nLayers; i++) {
        delete this->layersInput[i];
    }

    for(auto i = 0; i < this->nLayers - 1; i++) {
        delete this->weights[i];
    }

    delete this->layersInput;
    delete this->biases;
    delete this->weights;
}

void NeuralNet::init() {
    this->layerPos = 1;
    this->layersInput = new double*[this->nLayers];
    this->weights = new double*[this->nLayers - 1];

    for(auto i = 0; i < this->nLayers; i++) {
        int layerDim = this->network.at(i);
        if(i < this->nLayers - 1) ++layerDim;  // add 1 neuron for bias node; expect for output layer

        this->layersInput[i] = new double[layerDim];
        this->initNeurons(this->layersInput[i], layerDim);
        
        if(i > 0) {
            int prevLayerDim = this->network.at(i-1) + 1; // add one for the bias weight
            int numWeights = prevLayerDim * layerDim;
            this->weights[i-1] = new double[numWeights];
            this->initWeight(this->weights[i-1], numWeights);
        }
    }
    
    this->initBiases();
}

void NeuralNet::initBiases() {
    for(auto i = 0; i < (this->nLayers - 1); i++) {
        int layerDim = this->network.at(i) + 1; // add one for the bias
        this->layersInput[i][layerDim -1] = 1.0; // need to make it random
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

void NeuralNet::setBiases(double b[], const int n) {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        int layerDim = this->network.at(i) + 1; // add one for the bias
        this->layersInput[i][layerDim -1] = b[0];
    }
}

void NeuralNet::setWeights(double** w) {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        int layerDim = this->network.at(i+1);
        int prevLayerDim = this->network.at(i) + 1;
        memcpy(this->weights[i], w[i], sizeof(double) * layerDim * prevLayerDim);
    }
}

void NeuralNet::showW() {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        int layerDim = this->network.at(i+1);
        int prevLayerDim = this->network.at(i) + 1;
        cout << "L:" << i + 2 << endl; 
        for(auto j = 0; j < layerDim; j++) {
            cout << "  ";
            for(auto m = 0; m < prevLayerDim; m++) {
                double v = this->weights[i][prevLayerDim * j + m];
                cout << v << " ";
            }
            cout << endl; 
        }
    }
}

void NeuralNet::showN() {
    for(auto i = 0; i < this->nLayers; i++) {
        cout << "L:" << i + 1 << endl;
        cout << "  ";
        if(i == this->nLayers -1)  // output layer
            for(auto n = 0; n < this->network.at(i); n++) {
                cout << this->layersInput[i][n] << " ";
            }
        else
            // loop through all neuron + the bias neuron
            for(auto n = 0; n < this->network.at(i) + 1; n++) {
                cout << this->layersInput[i][n] << " ";
            }            
        cout << endl;
    }
}

void NeuralNet::forward() {
    int pos = this->layerPos++;
    int nrows = this->network.at(pos);
    int ncols = this->network.at(pos - 1) + 1; // add one for the bias neuron

    double* v = this->dotProduct(this->layersInput[pos - 1], this->weights[pos - 1], nrows, ncols);
    memcpy(this->layersInput[pos], v, sizeof(double) * nrows);
}

void NeuralNet::train() {

}

void NeuralNet::feedForward(double* input) {
    int input_shape = this->network.at(0);
    memcpy(this->layersInput[0], input, sizeof(double) * input_shape);
    this->layerPos = 1;
    for(auto i = this->layerPos; i < this->nLayers; i++) {
        this->forward();
    }

    int nOut = this->network.back();
    for(auto i = 0; i < nOut; i++) {
        cout << this->layersInput[this->nLayers - 1][i] << endl;
    }
}

double NeuralNet::sigmoid(double& val) {
    return 1.0/(1 + exp(-val));
}

double* NeuralNet::dotProduct(double* X, double* W, const int nrows, const int ncols) {
    const int a = 1;
    vector<double> y = {};
    for(auto i = 0; i < nrows; i++) {
        double prod = 0.0;
        for(auto j = 0; j < ncols; j++) {
            double x = X[j];
            double w_ij = W[ncols * i + j];
            prod += x * w_ij;
        }

        y.push_back(this->sigmoid(prod));
    }

    return y.data();
}