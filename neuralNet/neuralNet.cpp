#include <iostream>
#include "neuralNet.hpp"
#include <cmath>

using namespace std;
NeuralNet::NeuralNet(vector<vector<double>>& input, vector<vector<double>>& targets,  vector<unsigned>& network, double learningRate) {
    if(network.size() < 3) {
        throw "In valid shape.";
    }
    this->sampleInputIdx = -1;
    this->learningRate = learningRate;

    this->data = input;
    this->targets = targets;
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
    this->layersError = new Layer[this->nLayers];
    this->layersWeights = new Layer[this->nLayers - 1];

    for(auto i = 0; i < this->nLayers; i++) {
        unsigned layerDim = this->network.at(i);
        
        if(i < this->nLayers - 1) ++layerDim;  // add 1 neuron for bias node; expect for output layer

        this->layersInput[i] = new double[layerDim];
        this->layersError[i] = new double[layerDim];

        this->initNeurons(this->layersInput[i], layerDim);
        this->initNeurons(this->layersError[i], layerDim);

        if(i > 0) {  // each layer have  a weight for each input
            unsigned prevLayerDim = this->network.at(i-1) + 1; // add one for the bias weight
            unsigned numWeights = prevLayerDim * layerDim;  
            this->layersWeights[i-1] = new double[numWeights]; // express 2D as 1D here.
            this->initWeight(this->layersWeights[i-1], numWeights);
        }
    }
    
    this->initBiases();
}

void NeuralNet::initBiases() {
    for(auto i = 0; i < (this->nLayers - 1); i++) {
        unsigned layerDim = this->network.at(i) + 1; // add one for the bias
        this->layersInput[i][layerDim -1] = 1.0; // need to make it random
    }
}

void NeuralNet::initNeurons(double* neurons, unsigned& dim) {
    for(auto i = 0; i < dim; i++) {
        neurons[i] = 0.0;  // need to randomize
    }
}

void NeuralNet::initWeight(double* w, unsigned& dim) {
    for(auto i = 0; i < dim; i++) {
        w[i] = 0.0; // need to randomize
    }
}

void NeuralNet::setBiases(double b[], const unsigned n) {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        unsigned layerDim = this->network.at(i) + 1; // add one for the bias
        this->layersInput[i][layerDim -1] = b[0];
    }
}

void NeuralNet::setWeights(double** w) {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        unsigned layerDim = this->network.at(i+1);
        unsigned prevLayerDim = this->network.at(i) + 1;
        memcpy(this->layersWeights[i], w[i], sizeof(double) * layerDim * prevLayerDim);
    }
}

void NeuralNet::showW() {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        unsigned layerDim = this->network.at(i+1);
        unsigned prevLayerDim = this->network.at(i) + 1;
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
    this->fetchInput();
    double err = this->feedForward();
    cout << "error: " << err << endl; 
    this->backPropagation();
    this->learn();
}

double NeuralNet::feedForward() {
    Layer input = this->data.at(this->sampleInputIdx).data();
    Layer targetOutput = this->targets.at(this->sampleInputIdx).data();
    unsigned numNeurons = this->network.at(0);
    Layer inputLayer = this->layersInput[0];
    memcpy(inputLayer, input, sizeof(double) * numNeurons);

    this->layerPos = 1;
    for(auto i = this->layerPos; i < this->nLayers; i++) {
        this->forward();
    }
    double E = this->totalError(targetOutput);
    return E;
}

void NeuralNet::forward() {
    unsigned pos = this->layerPos;
    unsigned nrows = this->network.at(pos);
    unsigned ncols = this->network.at(pos - 1) + 1; // add one for the bias neuron

    Layer X = this->layersInput[pos - 1];  // get previous layer
    Layer Y = this->layersInput[pos];   // get current layer
    Layer W = this->layersWeights[pos - 1]; // -1 remember that layer 1 don't have Ws on its input
    
    Layer y = this->dotProduct(X, W, nrows, ncols, true); // Y = W * X  + b
    memcpy(Y, y, sizeof(double) * nrows);  // update curr layer outputs
    ++this->layerPos;
}

void NeuralNet::backPropagation() {
    // out layer error
    unsigned sampleInputIdx = this->sampleInputIdx;
    this->backPropageError(this->nLayers-1, this->targets[sampleInputIdx].data());
    for(auto layerIndex = this->nLayers - 2; layerIndex > 0; layerIndex--) {
        this->backPropageError(layerIndex);
    }
}

void NeuralNet::backPropageError(unsigned layerIndex, double* target) {
    Layer layerActivations = this->layersInput[layerIndex];
    Layer layerErrors = this->layersError[layerIndex];

    unsigned layerDim = this->network.at(layerIndex);
    if(layerIndex == this->nLayers - 1) {
        if(target == nullptr) {
            throw "Error: null target";
        }
        for(auto i = 0; i < layerDim; i++) {
            double a = layerActivations[i];
            layerErrors[i] = a * (1 - a) * (target[i] - a);
            // for(auto i = 0; i < layerDim; i++) {
            //     cout << layerErrors[i] << "  ";
            // }
            // cout << endl;
        }
    } else {
        Layer preLayer = this->layersError[layerIndex + 1];
        Layer W = this->layersWeights[layerIndex];
        unsigned nrows = layerDim;
        unsigned ncols = this->network.at(layerIndex + 1);
        Layer e = this->dotProduct(preLayer, W, nrows, ncols);

        for(auto i = 0; i < layerDim; i++) {  // take into accout the activation derivative
            double a = layerActivations[i];
            layerErrors[i] = (a * (1 - a)) * e[i];
        }

        // for(auto i = 0; i < nrows; i++) {
        //     cout << layerErrors[i] << "  ";
        // }
        // cout << endl;
    }
}

void NeuralNet::learn() {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        unsigned layerDim = this->network.at(i+1);
        unsigned prevLayerDim = this->network.at(i) + 1;
        for(auto j = 0; j < layerDim; j++) {
            double grad = this->layersError[i+1][j];
            for(auto m = 0; m < prevLayerDim; m++) {
                double a = this->layersInput[i][m];
                double w  = this->layersWeights[i][prevLayerDim * j + m];
                this->layersWeights[i][prevLayerDim * j + m] = w + (this->learningRate) * grad * a;
            }
        }
    }
}

double* NeuralNet::dotProduct(double* X, double* W, const unsigned nrows, const unsigned ncols, bool tranfer) {
    const unsigned a = 1;
    vector<double> y = {};
    for(auto i = 0; i < nrows; i++) {
        double prod = 0.0;
        for(auto j = 0; j < ncols; j++) {
            double x = X[j];
            double w_ij = W[ncols * i + j];
            prod += x * w_ij;
        }
        if(tranfer) prod = sigmoid(prod);
        y.push_back(prod);
    }

    return y.data();
}

double NeuralNet::sigmoid(double& val) {
    return 1.0/(1 + exp(-val));
}

double NeuralNet::totalError(double* target) {
    Layer outputLayer = this->layersInput[this->nLayers - 1];
    unsigned outputDim = this->network.back();
    double error = 0.0;
    for(auto i = 0; i < outputDim; i++) {
        error += (1.0/2) * pow((target[i] - outputLayer[i]), 2);
    }
    return error;
}


void NeuralNet::fetchInput() { ++this->sampleInputIdx; }

