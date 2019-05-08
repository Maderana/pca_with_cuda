CXX := g++ -O3 -std=c++11

OBJS := lab3_io.o

all: pca

clean:
	rm -f pca $(OBJS)

pca: main_cuda.cu lab3_cuda.cu lab3_io.cu $(OBJS)
	nvcc -lm -std=c++11 $^ -o $@

.PHONY: all clean
