#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include "activations.h"
#include "data.h"

int main(int argc, char *argv[]) {
    Network *net = (Network *)malloc(sizeof(Network));

    if (!load_weights(net, "weights/mnist.dat")) {
        printf("Train first: ./train\n");
        return 1;
    }

    Dataset test = load_mnist(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    );

    int start = (argc >= 2) ? atoi(argv[1]) : 0;
    int count = (argc >= 2) ? 1 : 10;
    int show  = (argc >= 3) ? 1 : 0;

    printf("%-6s %-8s %-8s %-8s\n", "Index", "Truth", "Pred", "Result");
    printf("------ -------- -------- --------\n");

    for (int i = start; i < start + count && i < test.count; i++) {
        if (show) { print_image(test.images[i]); }

        int pred  = predict(net, test.images[i]);
        int truth = test.labels[i];
        printf("%-6d %-8d %-8d %s\n",
               i, truth, pred, pred == truth ? "correct" : "WRONG");

        if (show) {
            forward(net, test.images[i]);
            for (int d = 0; d < OUTPUT_N; d++)
                printf("  %d: %.1f%%%s\n", d, net->a3[d] * 100.0,
                       d == pred ? " <- predicted" : "");
        }
    }

    free_dataset(&test);
    free(net);
    return 0;
}