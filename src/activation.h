#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

/* ─── Sigmoid ─────────────────────────────────────── */
double sigmoid(double x);
double sigmoid_d(double x);   /* derivative */

/* ─── Softmax ─────────────────────────────────────── */
/* Takes raw output scores, writes probabilities to out[] */
void softmax(double *x, double *out, int n);

/* ─── ReLU ────────────────────────────────────────── */
double relu(double x);
double relu_d(double x);      /* derivative */

#endif