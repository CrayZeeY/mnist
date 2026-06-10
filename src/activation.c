#include <math.h>
#include "activations.h"

/* ─── Sigmoid ─────────────────────────────────────── */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_d(double x) {
    return x * (1.0 - x);
}

/* ─── ReLU ────────────────────────────────────────── */
/*
 * relu(x) = x  if x > 0
 *         = 0  if x <= 0
 */
double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double relu_d(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

/* ─── Softmax ─────────────────────────────────────── */
/*
 * Subtracts max before exp() to prevent overflow.
 * Mathematically identical, numerically safe.
 */
void softmax(double *x, double *out, int n) {
    double max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = exp(x[i] - max);
        sum += out[i];
    }

    for (int i = 0; i < n; i++)
        out[i] /= sum;
}