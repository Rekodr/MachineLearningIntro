#ifndef DATA_READER_HPP
#define DATA_READER_HPP

#include <vector>
#include <string>

using namespace std;
class DataReader {
private:
    vector<vector<double>> X;
    vector<vector<double>> Y;
    vector<double> outT;
    unsigned inputWidth;
    unsigned outputWidth;
    unsigned inputSize;

    string filePath;
    void read();

public:
    DataReader(const string& file_path, const pair<unsigned, unsigned>& shape, const unsigned outputWidth);
    vector<vector<double>> getX() { return this->X; }
    vector<vector<double>> getY() { return this->Y; }
};
#endif