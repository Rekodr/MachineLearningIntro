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
        delete this->layersWeights[i];
    }

    delete this->layersInput;
    delete this->layersWeights;
}

void NeuralNet::init() {
    this->layerPos = 1;
    this->layersInput = new Layer[this->nLayers];
    this->layersWeights = new Layer[this->nLayers - 1];

    for(auto i = 0; i < this->nLayers; i++) {
        int layerDim = this->network.at(i);
        
        if(i < this->nLayers - 1) ++layerDim;  // add 1 neuron for bias node; expect for output layer

        this->layersInput[i] = new double[layerDim];
        this->initNeurons(this->layersInput[i], layerDim);

        if(i > 0) {  // each layer have  a weight for each input
            int prevLayerDim = this->network.at(i-1) + 1; // add one for the bias weight
            int numWeights = prevLayerDim * layerDim;  
            this->layersWeights[i-1] = new double[numWeights]; // express 2D as 1D here.
            this->initWeight(this->layersWeights[i-1], numWeights);
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
        memcpy(this->layersWeights[i], w[i], sizeof(double) * layerDim * prevLayerDim);
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
                double v = this->layersWeights[i][prevLayerDim * j + m];
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

void NeuralNet::train() {

}

void NeuralNet::feedForward(double* data) {
    int numNeurons = this->network.at(0);
    Layer inputLayer = this->layersInput[0];
    memcpy(inputLayer, data, sizeof(double) * numNeurons);

    this->layerPos = 1;

    for(auto i = this->layerPos; i < this->nLayers; i++) {
        this->forward();
    }

    int nOut = this->network.back();
    for(auto i = 0; i < nOut; i++) {
        cout << this->layersInput[this->nLayers - 1][i] << endl;
    }
}

void NeuralNet::forward() {
    int pos = this->layerPos;
    int nrows = this->network.at(pos);
    int ncols = this->network.at(pos - 1) + 1; // add one for the bias neuron

    Layer X = this->layersInput[pos - 1];  // get previous layer
    Layer Y = this->layersInput[pos];   // get current layer
    Layer W = this->layersWeights[pos - 1]; // -1 remember that layer 1 don't have Ws on its input
    
    Layer y = this->dotProduct(X, W, nrows, ncols); // Y = W * X  + b
    memcpy(Y, y, sizeof(double) * nrows);  // update curr layer outputs
    ++this->layerPos;
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

double NeuralNet::sigmoid(double& val) {
    return 1.0/(1 + exp(-val));
}