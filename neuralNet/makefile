TARGET=neuralNet
CC=g++
DEPS = neuralNet.hpp
OBJ = main.o neuralNet.o

%.o: %.cpp $(DEPS)
	$(CC) -std=c++17 -c -o $@ $<

$(TARGET): $(OBJ)
	$(CC) -o $@ $^

.PHONY: clean

clean:
	rm *.o $(TARGET)