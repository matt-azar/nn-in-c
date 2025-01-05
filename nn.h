#ifndef NN_H
#define NN_H

#include <math.h>
#include <stdlib.h>

// Network parameters
#define IMAGE_SIZE 784  // Do not modify IMAGE_SIZE or OUTPUT_SIZE.
#define HIDDEN_SIZE1 32 // Increase size of hidden layers for increased accuracy
#define HIDDEN_SIZE2 64 // at the cost of increased runtime. Recommended range:
#define OUTPUT_SIZE 10  // ~ (10, 150) for reasonable accuracy and runtime.

typedef struct {
    size_t input_size;
    size_t hidden_size1;
    size_t hidden_size2;
    size_t output_size;

    double *weights1;
    double *weights2;
    double *weights3;
    double *biases1;
    double *biases2;
    double *biases3;

    double *hidden1;
    double *hidden2;
    double *output;
} Network;

// Function prototypes
void initialize_network(Network *net, int input_size, int hidden_size1, int hidden_size2, int output_size);
void free_network(Network *net);

void initialize_weights(double *weights, int size, int inputs);
void initialize_biases(double *biases, int size);

double relu(double x);
double relu_derivative(double x);
void softmax(double *output, int size);

void forward_pass(Network *net, double *input, double dropout_rate, int is_training);
void backpropagate(Network *net, double *input, unsigned char target_label, double learning_rate);

double compute_loss(Network *net, unsigned char target_label);
double dropout(double x, double dropout_rate);
int predict(Network *net);

void save_network(const char *filename, Network *net);
void load_network(const char *filename, Network *net);

#endif // NN_H
