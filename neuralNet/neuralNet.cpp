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
    this->biases = new double[this->nLayers - 1]; 
    this->weights = new double*[this->nLayers - 1];

    this->initBiases(this->biases);
    
    for(auto i = 0; i < this->nLayers; i++) {
        int layerDim = this->network.at(i) ;
        this->layersInput[i] = new double[layerDim];
        this->initNeurons(this->layersInput[i], layerDim);
        if(i > 0) {
            int prevLayerDim = this->network.at(i-1) + 1; // add one for the bias weight
            int wDim = prevLayerDim * layerDim;
            this->weights[i-1] = new double[wDim];
            this->initWeight(this->weights[i-1], wDim);
        }
    }
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

void NeuralNet::setBiases(double b[], const int n) {
    memcpy(this->biases, b, sizeof(double) * (this->nLayers - 1));
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

void NeuralNet::showB() {
    for(auto i = 0; i < this->nLayers -1; i++) {
        cout << "  " << this->biases[i] << endl;
    }
}

void NeuralNet::showN() {
    for(auto i = 0; i < this->nLayers; i++) {
        cout << "L:" << i + 1 << endl;
        cout << "  ";
        for(auto n = 0; n < this->network.at(i); n++) {
            cout << this->layersInput[i][n] << " ";
        }
        cout << endl;
    }
}

void NeuralNet::forward() {
    int pos = this->layerPos++;
    int nrows = this->network.at(pos);
    int ncols = this->network.at(pos - 1) + 1; // add one for the bias neuron
    double b = this->biases[pos - 1];
    double* v = this->yCpu(this->layersInput[pos - 1], this->weights[pos - 1], b, nrows, ncols);
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

double* NeuralNet::yCpu(double* X, double* W, double b,const int nrows, const int ncols) {
    const int a = 1;
    vector<double> y = {};
    for(auto i = 0; i < nrows; i++) {
        double prod = 0.0;
        for(auto j = 0; j < ncols; j++) {
            double val = 0.0;
            if(j < ncols - 1) {
                val = X[j]; 
            } else {
                val = b; 
            }
            double wi = W[ncols * i + j];
            prod += val * wi;
        }
        y.push_back(this->sigmoid(prod));
    }

    return y.data();
}