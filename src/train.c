#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "network.h"
#include "activations.h"
#include "data.h"

#define EPOCHS     10
#define BATCH_SIZE 32

static void shuffle(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

static double evaluate(Network *net, Dataset *d) {
    int correct = 0;
    for (int i = 0; i < d->count; i++)
        if (predict(net, d->images[i]) == d->labels[i]) correct++;
    return (double)correct / d->count * 100.0;
}

int main() {
    srand(time(NULL));

    Dataset train = load_mnist(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte"
    );
    Dataset test = load_mnist(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    );

    if (train.count == 0 || test.count == 0) return 1;

    Network *net = (Network *)malloc(sizeof(Network));
    init_network(net);

    int *indices = (int *)malloc(train.count * sizeof(int));
    for (int i = 0; i < train.count; i++) indices[i] = i;

    printf("%-6s %-12s %-12s %-12s\n",
           "Epoch", "Loss", "Train Acc", "Test Acc");
    printf("------ ------------ ------------ ------------\n");

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        shuffle(indices, train.count);

        double total_loss = 0.0;
        int batches = 0;

        for (int b = 0; b < train.count; b += BATCH_SIZE) {
            int end = b + BATCH_SIZE;
            if (end > train.count) end = train.count;

            double batch_loss = 0.0;
            for (int s = b; s < end; s++) {
                int idx = indices[s];
                forward(net, train.images[idx]);

                double prob = net->a3[train.labels[idx]];
                if (prob < 1e-10) prob = 1e-10;
                batch_loss -= log(prob);

                backward(net, train.labels[idx]);
            }
            total_loss += batch_loss / (end - b);
            batches++;
        }

        printf("%-6d %-12.4f %-12.2f %-12.2f\n",
               epoch + 1,
               total_loss / batches,
               evaluate(net, &train),
               evaluate(net, &test));

        save_weights(net, "weights/mnist.dat");
    }

    free_dataset(&train);
    free_dataset(&test);
    free(indices);
    free(net);
    return 0;
}