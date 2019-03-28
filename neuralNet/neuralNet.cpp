#include "neuralNet.hpp"

using namespace std;
NeuralNet::NeuralNet(vector<double> data, vector<int>& shape, int nLayers) {
    if(shape.size() < 3) {
        throw "In valid shape.";
    }

    this -> shape.assign(shape.begin(), shape.end()); 
    this -> nHiddedlayers = nLayers;
    // int array_dim = in_size* layers;
    // this -> hlayers = shared_ptr<double>(new double(array_dim));
    // this -> weights = shared_ptr<double>(new double(array_dim));
    // this -> biases = shared_ptr<double>(new double(layers + 1));

    this -> inputs = shared_ptr<double>(new double(data.size()));
    this -> output = shared_ptr<double>(new double(1));
}

NeuralNet::~NeuralNet() {}

void NeuralNet::init(vector<double>& data) {

    
    if(this -> shape.size() == 3) {
    }
}