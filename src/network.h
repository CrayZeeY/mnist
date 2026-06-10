#ifndef NETWORK_H
#define NETWORK_H

/* ─── Network architecture ────────────────────────── */
#define INPUT_N   784    /* 28x28 pixels */
#define HIDDEN1_N 128    /* first hidden layer  */
#define HIDDEN2_N 64     /* second hidden layer */
#define OUTPUT_N  10     /* digits 0-9 */
#define LR        0.01   /* learning rate */

/* ─── Network struct ──────────────────────────────── */
typedef struct {
    /* Weights */
    double W1[INPUT_N][HIDDEN1_N];
    double W2[HIDDEN1_N][HIDDEN2_N];
    double W3[HIDDEN2_N][OUTPUT_N];

    /* Biases */
    double b1[HIDDEN1_N];
    double b2[HIDDEN2_N];
    double b3[OUTPUT_N];

    /* Layer outputs — saved during forward pass for backprop */
    double a0[INPUT_N];      /* input layer  */
    double a1[HIDDEN1_N];    /* hidden layer 1 */
    double a2[HIDDEN2_N];    /* hidden layer 2 */
    double a3[OUTPUT_N];     /* output (probabilities after softmax) */
    double z3[OUTPUT_N];     /* output (raw scores before softmax) */

} Network;

/* ─── Function declarations ───────────────────────── */
void  init_network(Network *net);
void  forward(Network *net, double *input);
void  backward(Network *net, int true_label);
void  save_weights(Network *net, const char *filename);
int   load_weights(Network *net, const char *filename);
int   predict(Network *net, double *input);

#endif