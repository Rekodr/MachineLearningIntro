TARGET=NN
CC=g++
DEPS = neuralNet.hpp
OBJ = main.o neuralNet.o
CFLAGS = -g

%.o: %.cpp $(DEPS)
	$(CC) -std=c++17 -c -o $@ $< $(CFLAGS)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm *.o $(TARGET)