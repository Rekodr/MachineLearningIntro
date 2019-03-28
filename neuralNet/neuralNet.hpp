#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <memory>
#include <vector>

using namespace std;

class NeuralNet {
private:
    int nHiddedlayers;
    shared_ptr<double> inputs;
    shared_ptr<double> hlayers;
    shared_ptr<double> output;
    shared_ptr<double> weights;
    shared_ptr<double> biases;
    vector<int> shape;
public:
    NeuralNet(vector<double> data, vector<int>& shape, int nLayers=1);
    ~NeuralNet();
    void init(vector<double>& data);
    int getnLayers() { return this -> nHiddedlayers;};
};
#endif