#ifndef NN_H
#define NN_H

#include <math.h>
#include <stdlib.h>

// Network parameters
#define IMAGE_SIZE 784
#define HIDDEN_SIZE1 12
#define HIDDEN_SIZE2 12
#define OUTPUT_SIZE 10

void initialize_weights(double *weights, int size, int inputs);
void initialize_biases(double *biases, int size);

double relu(double x);
double relu_derivative(double x);
void softmax(double *output, int size);

void forward_pass(double *input, double *weights1, double *weights2, double *weights3,
                  double *biases1, double *biases2, double *biases3,
                  double *hidden1, double *hidden2, double *output, 
                  double dropout_rate, int is_training);

void backpropagate(double *input, double *weights1, double *weights2, double *weights3,
                   double *biases1, double *biases2, double *biases3,
                   double *hidden1, double *hidden2, double *output,
                   unsigned char target_label, double learning_rate);

double compute_loss(double *output, unsigned char target_label);
double dropout(double x, double dropout_rate);
int predict(double *output);

void save_network(const char *filename, double *weights1, double *weights2, double *weights3,
                  double *biases1, double *biases2, double *biases3);
void load_network(const char *filename, double *weights1, double *weights2, double *weights3,
                  double *biases1, double *biases2, double *biases3);

#endif // NN_H
