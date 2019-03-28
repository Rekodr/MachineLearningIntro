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

void NeuralNet::showW() {
    cout << "W" << endl;
    for(auto i = 0; i < this->nLayers - 1; i++) {
        int layerDim = this->network.at(i+1);
        cout << "L:" << i + 2 << endl; 
        cout << "  ";
        int prevLayerDim = this->network.at(i) + 1;
        for(auto j = 0; j < layerDim; j++) {
            for(auto m = 0; m < prevLayerDim; m++) {
                double v = this->weights[i][prevLayerDim * j + m];
                cout << v << " ";
            }
            cout << endl; 
        }
    }
}

void NeuralNet::showB() {
    cout << "B" << endl;
    for(auto i = 0; i < this->nLayers -1; i++) {
        cout << "  " << this->biases[i] << endl;
    }
}

void NeuralNet::showN() {
    cout << "X" << endl;
    for(auto i = 0; i < this->nLayers; i++) {
        cout << "L:" << i + 1 << endl;
        cout << "  ";
        for(auto n = 0; n < this->network.at(i); n++) {
            cout << this->layersInput[i][n] << " ";
        }
        cout << endl;
    }
}