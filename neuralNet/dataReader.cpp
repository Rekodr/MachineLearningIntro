#include "dataReader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

DataReader::DataReader(const string& file_path, const pair<unsigned, unsigned>& shape, const unsigned outputWidth) {
    this->inputWidth = shape.second;
    this->inputSize = shape.first;
    this->filePath = file_path;
    this->outputWidth = outputWidth;

    for(auto i = 0; i < outputWidth; i++) {
        this->outT.push_back(0);
    }
    this->read();
}

void DataReader::read() {
    ifstream in(this->filePath);
    if(!in.is_open()) {
        throw "file not found";
    }

    string line;
    for(auto i = 0; i < this->inputSize; i++) {
        getline(in, line);
        stringstream ss(line);
        string x;
        vector<double> row{};
        while(getline(ss, x, ',')) {
            double val = stod(x);
            row.push_back(val);
        }
        const unsigned y =  (unsigned)(row.back());

        row.pop_back();
        if(row.size() < this->inputWidth) {
            throw "Bad data format";
        }
        
        this->X.push_back(row);

        vector<double> o{this->outT} ;
        o.at(y) = 1;
        // cout << y << " ";
        // copy(o.begin(), o.end(), ostream_iterator<double>(std::cout, " "));
        // cout << endl;
        this->Y.push_back(o);
    }

    in.close();
}