TARGET=neuralNet
CC=g++
DEPS = neuralNet.hpp
OBJ = main.o neuralNet.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $<

$(TARGET): $(OBJ)
	$(CC) -o $@ $^

.PHONY: clean

clean:
	rm *.o $(Name)