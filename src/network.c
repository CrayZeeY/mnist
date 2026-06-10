#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "activations.h"

/* ─── Xavier initialization ───────────────────────── */
static double xavier(int fan_in, int fan_out) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    return ((double)rand() / RAND_MAX) * 2.0 * limit - limit;
}

/* ─── Init network ────────────────────────────────── */
void init_network(Network *net) {
    srand(time(NULL));

    for (int i = 0; i < INPUT_N; i++)
        for (int j = 0; j < HIDDEN1_N; j++)
            net->W1[i][j] = xavier(INPUT_N, HIDDEN1_N);

    for (int i = 0; i < HIDDEN1_N; i++)
        for (int j = 0; j < HIDDEN2_N; j++)
            net->W2[i][j] = xavier(HIDDEN1_N, HIDDEN2_N);

    for (int i = 0; i < HIDDEN2_N; i++)
        for (int j = 0; j < OUTPUT_N; j++)
            net->W3[i][j] = xavier(HIDDEN2_N, OUTPUT_N);

    for (int j = 0; j < HIDDEN1_N; j++) net->b1[j] = 0.0;
    for (int j = 0; j < HIDDEN2_N; j++) net->b2[j] = 0.0;
    for (int j = 0; j < OUTPUT_N;  j++) net->b3[j] = 0.0;
}

/* ─── Forward pass ────────────────────────────────── */
void forward(Network *net, double *input) {

    for (int i = 0; i < INPUT_N; i++)
        net->a0[i] = input[i];

    /* Input -> Hidden1 (ReLU) */
    for (int j = 0; j < HIDDEN1_N; j++) {
        double z = net->b1[j];
        for (int i = 0; i < INPUT_N; i++)
            z += net->a0[i] * net->W1[i][j];
        net->a1[j] = relu(z);
    }

    /* Hidden1 -> Hidden2 (ReLU) */
    for (int j = 0; j < HIDDEN2_N; j++) {
        double z = net->b2[j];
        for (int i = 0; i < HIDDEN1_N; i++)
            z += net->a1[i] * net->W2[i][j];
        net->a2[j] = relu(z);
    }

    /* Hidden2 -> Output (Softmax) */
    for (int j = 0; j < OUTPUT_N; j++) {
        net->z3[j] = net->b3[j];
        for (int i = 0; i < HIDDEN2_N; i++)
            net->z3[j] += net->a2[i] * net->W3[i][j];
    }
    softmax(net->z3, net->a3, OUTPUT_N);
}

/* ─── Backward pass ───────────────────────────────── */
void backward(Network *net, int true_label) {

    /* Output delta — softmax + cross-entropy simplifies to this */
    double delta3[OUTPUT_N];
    for (int j = 0; j < OUTPUT_N; j++)
        delta3[j] = net->a3[j] - (j == true_label ? 1.0 : 0.0);

    /* Hidden2 delta */
    double delta2[HIDDEN2_N];
    for (int j = 0; j < HIDDEN2_N; j++) {
        double err = 0.0;
        for (int k = 0; k < OUTPUT_N; k++)
            err += delta3[k] * net->W3[j][k];
        delta2[j] = err * relu_d(net->a2[j]);
    }

    /* Hidden1 delta */
    double delta1[HIDDEN1_N];
    for (int j = 0; j < HIDDEN1_N; j++) {
        double err = 0.0;
        for (int k = 0; k < HIDDEN2_N; k++)
            err += delta2[k] * net->W2[j][k];
        delta1[j] = err * relu_d(net->a1[j]);
    }

    /* Update W3, b3 */
    for (int i = 0; i < HIDDEN2_N; i++)
        for (int j = 0; j < OUTPUT_N; j++)
            net->W3[i][j] -= LR * delta3[j] * net->a2[i];
    for (int j = 0; j < OUTPUT_N; j++)
        net->b3[j] -= LR * delta3[j];

    /* Update W2, b2 */
    for (int i = 0; i < HIDDEN1_N; i++)
        for (int j = 0; j < HIDDEN2_N; j++)
            net->W2[i][j] -= LR * delta2[j] * net->a1[i];
    for (int j = 0; j < HIDDEN2_N; j++)
        net->b2[j] -= LR * delta2[j];

    /* Update W1, b1 */
    for (int i = 0; i < INPUT_N; i++)
        for (int j = 0; j < HIDDEN1_N; j++)
            net->W1[i][j] -= LR * delta1[j] * net->a0[i];
    for (int j = 0; j < HIDDEN1_N; j++)
        net->b1[j] -= LR * delta1[j];
}

/* ─── Predict ─────────────────────────────────────── */
int predict(Network *net, double *input) {
    forward(net, input);
    int best = 0;
    for (int j = 1; j < OUTPUT_N; j++)
        if (net->a3[j] > net->a3[best]) best = j;
    return best;
}

/* ─── Save / Load weights ─────────────────────────── */
void save_weights(Network *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) { printf("Error: cannot open %s\n", filename); return; }
    fwrite(net->W1, sizeof(net->W1), 1, f);
    fwrite(net->W2, sizeof(net->W2), 1, f);
    fwrite(net->W3, sizeof(net->W3), 1, f);
    fwrite(net->b1, sizeof(net->b1), 1, f);
    fwrite(net->b2, sizeof(net->b2), 1, f);
    fwrite(net->b3, sizeof(net->b3), 1, f);
    fclose(f);
    printf("Weights saved to %s\n", filename);
}

int load_weights(Network *net, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { printf("Error: cannot open %s\n", filename); return 0; }
    fread(net->W1, sizeof(net->W1), 1, f);
    fread(net->W2, sizeof(net->W2), 1, f);
    fread(net->W3, sizeof(net->W3), 1, f);
    fread(net->b1, sizeof(net->b1), 1, f);
    fread(net->b2, sizeof(net->b2), 1, f);
    fread(net->b3, sizeof(net->b3), 1, f);
    fclose(f);
    printf("Weights loaded from %s\n", filename);
    return 1;
}