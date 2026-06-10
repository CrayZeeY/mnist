CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -Isrc
LIBS    = -lm

COMMON  = src/activations.c src/network.c src/data.c

all: train predict

train: $(COMMON) src/train.c
	$(CC) $(CFLAGS) -o train $(COMMON) src/train.c $(LIBS)

predict: $(COMMON) src/predict.c
	$(CC) $(CFLAGS) -o predict $(COMMON) src/predict.c $(LIBS)

data:
	cd data && \
	wget -q --show-progress http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz && \
	wget -q --show-progress http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz && \
	wget -q --show-progress http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  && \
	wget -q --show-progress http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz  && \
	gunzip *.gz && \
	echo "MNIST data ready."

clean:
	rm -f train predict

cleanall: clean
	rm -f weights/*.dat

.PHONY: all clean cleanall data