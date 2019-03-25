#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <memory>

using namespace std;
class NeuralNet {
private:
    int nHiddedlayers;
    shared_ptr<double> inputs;
    shared_ptr<double> weights;
    shared_ptr<double> biases;
public:
    NeuralNet(int layers=2);
    ~NeuralNet();
    int getnLayers() { return this -> nHiddedlayers;};
};
#endif