#include <iostream>
#include "neuralNet.hpp"
#include <cmath>
#include <time.h>

using namespace std;

NeuralNet::NeuralNet(vector<vector<double>>& input, vector<vector<double>>& targets,  vector<unsigned int>& network,
    double learningRate, unsigned int epochs, unsigned int miniBatchSize) {

    srand (time(NULL));   

    if(network.size() < 3) {
        throw "In valid shape.";
    }
    this->sampleInputIdx = -1;
    this->learningRate = learningRate;
    this->epochs = epochs;
    
    this->data = input;
    this->targets = targets;
    this->network.assign(network.begin(), network.end()); 
    this->nLayers = (unsigned int)network.size();

    if(miniBatchSize > data.size()) 
        this->miniBatchSize = (unsigned int) data.size();
    else 
        this->miniBatchSize = miniBatchSize;
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
        unsigned int layerDim = this->network.at(i);
        
        if(i < this->nLayers - 1) ++layerDim;  // add 1 neuron for bias node; expect for output layer

        this->layersInput[i] = new double[layerDim];
        this->layersError[i] = new double[layerDim];

        this->initNeurons(this->layersInput[i], layerDim);
        this->initNeurons(this->layersError[i], layerDim);
        if(i < this->nLayers - 1) --layerDim;

        if(i > 0) {  // each layer have  a weight for each input
            unsigned int prevLayerDim = this->network.at(i-1) + 1; // add one for the bias weight
            unsigned int numWeights = prevLayerDim * layerDim;  
            this->layersWeights[i-1] = new double[numWeights]; // express 2D as 1D here.
            this->initWeight(this->layersWeights[i-1], numWeights, prevLayerDim);
        }
    }
    
    this->initBiases();
}

void NeuralNet::train() {
    double err = 0.0;
    for(auto i = 0; i < this->epochs; i++) {
        this->sampleInputIdx = 0;
        while(this->sampleInputIdx < this->data.size()) {
            for(auto j = 0; j < this->miniBatchSize && this->sampleInputIdx < this->data.size(); j++) {
            
                err = this->feedForward();
                this->backPropagation();
                this->fetchInput();
            }

            /* if the number of sample per min batch is > 1,
             compute the avg of deltas
            */
            if(this->miniBatchSize > 1) {
                for(auto i = 0; i < this->nLayers; i++) {
                    unsigned int layerDim = this->network.at(i);
                    if(i < this->nLayers - 1) ++layerDim; 
                    this->avgErrors(this->layersError[i], layerDim, this->miniBatchSize);
                }        
            }

            this->learn();
            this->clearErrors();
        }
        cout << "error: " << err << endl;
    }
}

void NeuralNet::test(vector<vector<double>>& input, vector<vector<double>>& targets) {
    unsigned int correct = 0;
    for(auto i = 0; i < input.size(); i++) {
        auto t = targets.at(i);
        unsigned int c = this->classify(input.at(i));
        if(t.at(c) == 1) {
            ++correct;
        }
    }
    double acc = (double) correct / input.size();
    cout << "Acc: " << acc << endl;

}

unsigned int NeuralNet::classify(vector<double>& X) {
    Layer input = X.data();
    unsigned int numNeurons = this->network.at(0);
    Layer inputLayer = this->layersInput[0];
    memcpy(inputLayer, input, sizeof(double) * numNeurons);

    this->layerPos = 1;
    for(auto i = this->layerPos; i < this->nLayers; i++) {
        this->forward();
    }

    Layer output = this->layersInput[this->nLayers -1]; 
    double max = 0.0;
    unsigned int index = 0;
    for(auto i = 0; i < this->network.back(); i++) {
        if(output[i] > max) {
            max = output[i];
            index = i;
        }
    }
    return index;
}

double NeuralNet::feedForward() {
    Layer input = this->data.at(this->sampleInputIdx).data();
    Layer targetOutput = this->targets.at(this->sampleInputIdx).data();
    unsigned int numNeurons = this->network.front();
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
    unsigned int pos = this->layerPos;
    unsigned int nrows = this->network.at(pos);
    unsigned int ncols = this->network.at(pos - 1) + 1; // add one for the bias neuron

    Layer X = this->layersInput[pos - 1];  // get previous layer
    Layer Y = this->layersInput[pos];   // get current layer
    Layer W = this->layersWeights[pos - 1]; // -1 remember that layer 1 don't have Ws on its input
    Layer y = nullptr;
    if(this->layerPos < this->nLayers - 1 || this->network.back() < 2) {
        y = this->dotProduct(X, W, nrows, ncols, "sig"); // Y = W * X  + b
    } else {
        y = this->dotProduct(X, W, nrows, ncols); // Y = W * X  + b

        double sum = 0.0;
        for(auto i = 0; i < this->network.back(); i++) {
            double a = y[i];

            sum += exp(a);

        }

        for(auto i = 0; i < this->network.back(); i++) {
            double tmp = exp(y[i]);     
            double a = tmp / sum;
            y[i] = a;
        }
    }
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

void NeuralNet::backPropageError(unsigned int layerIndex, double* target) {
    Layer layerActivations = this->layersInput[layerIndex];
    Layer layerErrors = this->layersError[layerIndex];

    unsigned int layerDim = this->network.at(layerIndex);
    if(layerIndex == this->nLayers - 1) {
        if(target == nullptr) {
            throw "Error: null target";
        }
        for(auto i = 0; i < layerDim; i++) {
            double a = layerActivations[i];
            double y = target[i];
            layerErrors[i] += a * (1 - a) * (y - a);
        }
    } else {
        Layer preLayer = this->layersError[layerIndex + 1];
        Layer W = this->layersWeights[layerIndex];
        unsigned int nrows = layerDim;
        unsigned int ncols = this->network.at(layerIndex + 1);
        Layer e = this->dotProduct(preLayer, W, nrows, ncols);

        for(auto i = 0; i < layerDim; i++) {  // take into accout the activation derivative
            double a = layerActivations[i];
            layerErrors[i] += (a * (1 - a)) * e[i];
        }
    }
}

void NeuralNet::learn() {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        unsigned int layerDim = this->network.at(i+1);
        unsigned int prevLayerDim = this->network.at(i) + 1;
        for(auto j = 0; j < layerDim; j++) {
            double grad = this->layersError[i+1][j];
            for(auto m = 0; m < prevLayerDim; m++) {
                double a = this->layersInput[i][m];
                double w  = this->layersWeights[i][prevLayerDim * j + m];
                double new_w = w + (this->learningRate) * grad * a;
                this->layersWeights[i][prevLayerDim * j + m] = new_w;
            }
        }
    }
}

double* NeuralNet::dotProduct(Layer X, Layer W, const unsigned int nrows, const unsigned int ncols, string tranfer) {
    double x, w_ij;
    vector<double> y{};
    double sigma = 0.0;
    for(auto i = 0; i < nrows; i++) {
        sigma = 0.0;
        for(auto j = 0; j < ncols; j++) {
            x = X[j];
            w_ij = W[ncols * i + j];
            sigma += x * w_ij;
        }
        if(tranfer == "sig") {
            sigma = sigmoid(sigma);
        }
        if(isnan(sigma) || sigma > 10000) {
            throw "error";
        }
        y.push_back(sigma);
    }

    return y.data();
}

double NeuralNet::sigmoid(double& val) {
    return 1.0/(1 + exp(-val * 1.0));
}

double NeuralNet::totalError(Layer target) {
    Layer outputLayer = this->layersInput[this->nLayers - 1];
    unsigned int outputDim = this->network.back();
    double error = 0.0;
    for(auto i = 0; i < outputDim; i++) {
        double y = target[i];
        double a = outputLayer[i];
        error += pow((a - y), 2);
    }
    return error;
}

void NeuralNet::clearErrors() {
    for(auto i = 0; i < this->nLayers; i++) {
        unsigned int layerDim = this->network.at(i);
        if(i < this->nLayers - 1) ++layerDim; 
        this->initNeurons(this->layersError[i], layerDim);
    }
}

void NeuralNet::avgErrors(Layer deltas, unsigned int& dim, unsigned int N) {
    for(auto i = 0; i < dim; i++) {
        deltas[i] = deltas[i] / N;  // need to randomize
    }
}

void NeuralNet::fetchInput() { ++this->sampleInputIdx; }

void NeuralNet::initBiases() {
    for(auto i = 0; i < (this->nLayers - 1); i++) {
        unsigned int layerDim = this->network.at(i) + 1; // add one for the bias
        this->layersInput[i][layerDim -1] = 1.0; // need to make it random
    }
}

void NeuralNet::initNeurons(double* neurons, unsigned int& dim) {
    for(auto i = 0; i < dim; i++) {
        neurons[i] = 0.0;  // need to randomize
    }
}

double fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void NeuralNet::initWeight(double* w, unsigned int& dim, unsigned int& nNodes) {
    unsigned int n = nNodes - 1;
    for(auto i = 0; i < dim; i++) {
        double wi = fRand(-1.0/sqrt(n), 1.0/sqrt(n));
        while(wi == 0) {
            wi = fRand(-1.0/sqrt(n), 1.0/sqrt(n));
        }
        w[i] = wi;
        if(isnan(wi)) {
            cout << "OKKKKKKKKKKKKKKKKKKKKKKKKKKKKK";
        }
    }
}

void NeuralNet::setBiases(double b[], const unsigned int n) {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        unsigned int layerDim = this->network.at(i) + 1; // add one for the bias
        this->layersInput[i][layerDim -1] = b[0];
    }
}

void NeuralNet::setWeights(Layers w) {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        unsigned int layerDim = this->network.at(i+1);
        unsigned int prevLayerDim = this->network.at(i) + 1;
        memcpy(this->layersWeights[i], w[i], sizeof(double) * layerDim * prevLayerDim);
    }
}

void NeuralNet::showW() {
    for(auto i = 0; i < this->nLayers - 1; i++) {
        unsigned int layerDim = this->network.at(i+1);
        unsigned int prevLayerDim = this->network.at(i) + 1;
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
