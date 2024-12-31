#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Initialize the network
void initialize_network(Network *net, int input_size, int hidden_size1, int hidden_size2, int output_size) {
    net->input_size = input_size;
    net->hidden_size1 = hidden_size1;
    net->hidden_size2 = hidden_size2;
    net->output_size = output_size;

    net->weights1 = malloc(input_size * hidden_size1 * sizeof(double));
    net->weights2 = malloc(hidden_size1 * hidden_size2 * sizeof(double));
    net->weights3 = malloc(hidden_size2 * output_size * sizeof(double));
    net->biases1 = malloc(hidden_size1 * sizeof(double));
    net->biases2 = malloc(hidden_size2 * sizeof(double));
    net->biases3 = malloc(output_size * sizeof(double));
    net->hidden1 = malloc(hidden_size1 * sizeof(double));
    net->hidden2 = malloc(hidden_size2 * sizeof(double));
    net->output = malloc(output_size * sizeof(double));
}

// Free the network resources
void free_network(Network *net) {
    free(net->weights1);
    free(net->weights2);
    free(net->weights3);
    free(net->biases1);
    free(net->biases2);
    free(net->biases3);
    free(net->hidden1);
    free(net->hidden2);
    free(net->output);
}

// Initialize weights using Xavier initialization
void initialize_weights(double *weights, int size, int inputs) {
    double limit = sqrt(6.0 / inputs);
    for (int i = 0; i < size; ++i) {
        weights[i] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
    }
}

// Initialize biases to small random values
void initialize_biases(double *biases, int size) {
    for (int i = 0; i < size; ++i) {
        biases[i] = ((double)rand() / RAND_MAX) / 100;
    }
}

// ReLU activation function
double relu(double x) {
    return x > 0 ? x : 0;
}

// Derivative of ReLU activation function
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Softmax activation function
void softmax(double *output, int size) {
    double max = output[0], sum = 0.0;
    for (int i = 1; i < size; ++i) {
        if (output[i] > max)
            max = output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i] = exp(output[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

// Forward pass through the network
void forward_pass(Network *net, double *input, double dropout_rate, int is_training) {
    // First hidden layer
    for (int i = 0; i < net->hidden_size1; ++i) {
        net->hidden1[i] = net->biases1[i];
        for (int j = 0; j < net->input_size; ++j) {
            net->hidden1[i] += input[j] * net->weights1[i * net->input_size + j];
        }
        net->hidden1[i] = relu(net->hidden1[i]);
        if (is_training) {
            net->hidden1[i] = dropout(net->hidden1[i], dropout_rate);
        }
    }

    // Second hidden layer
    for (int i = 0; i < net->hidden_size2; ++i) {
        net->hidden2[i] = net->biases2[i];
        for (int j = 0; j < net->hidden_size1; ++j) {
            net->hidden2[i] += net->hidden1[j] * net->weights2[i * net->hidden_size1 + j];
        }
        net->hidden2[i] = relu(net->hidden2[i]);
        if (is_training) {
            net->hidden2[i] = dropout(net->hidden2[i], dropout_rate);
        }
    }

    // Output layer
    for (int i = 0; i < net->output_size; ++i) {
        net->output[i] = net->biases3[i];
        for (int j = 0; j < net->hidden_size2; ++j) {
            net->output[i] += net->hidden2[j] * net->weights3[i * net->hidden_size2 + j];
        }
    }

    softmax(net->output, net->output_size);
}

// Backpropagation to update weights and biases
void backpropagate(Network *net, double *input, unsigned char target_label, double learning_rate) {
    double target[net->output_size];
    for (int i = 0; i < net->output_size; ++i)
        target[i] = 0.0;
    target[target_label] = 1.0;

    double output_error[net->output_size], output_delta[net->output_size];
    double hidden2_error[net->hidden_size2], hidden2_delta[net->hidden_size2];
    double hidden1_error[net->hidden_size1], hidden1_delta[net->hidden_size1];

    // Output layer errors and deltas
    for (int i = 0; i < net->output_size; ++i) {
        output_error[i] = target[i] - net->output[i];
        output_delta[i] = output_error[i];
        net->biases3[i] += learning_rate * output_delta[i];
    }

    // Hidden layer 2 errors and deltas
    for (int i = 0; i < net->hidden_size2; ++i) {
        hidden2_error[i] = 0.0;
        for (int j = 0; j < net->output_size; ++j) {
            hidden2_error[i] += output_delta[j] * net->weights3[j * net->hidden_size2 + i];
        }
        hidden2_delta[i] = hidden2_error[i] * relu_derivative(net->hidden2[i]);
        net->biases2[i] += learning_rate * hidden2_delta[i];
    }

    // Hidden layer 1 errors and deltas
    for (int i = 0; i < net->hidden_size1; ++i) {
        hidden1_error[i] = 0.0;
        for (int j = 0; j < net->hidden_size2; ++j) {
            hidden1_error[i] += hidden2_delta[j] * net->weights2[j * net->hidden_size1 + i];
        }
        hidden1_delta[i] = hidden1_error[i] * relu_derivative(net->hidden1[i]);
        net->biases1[i] += learning_rate * hidden1_delta[i];
    }

    // Update weights
    for (int i = 0; i < net->output_size; ++i) {
        for (int j = 0; j < net->hidden_size2; ++j) {
            net->weights3[i * net->hidden_size2 + j] += learning_rate * output_delta[i] * net->hidden2[j];
        }
    }

    for (int i = 0; i < net->hidden_size2; ++i) {
        for (int j = 0; j < net->hidden_size1; ++j) {
            net->weights2[i * net->hidden_size1 + j] += learning_rate * hidden2_delta[i] * net->hidden1[j];
        }
    }

    for (int i = 0; i < net->hidden_size1; ++i) {
        for (int j = 0; j < net->input_size; ++j) {
            net->weights1[i * net->input_size + j] += learning_rate * hidden1_delta[i] * input[j];
        }
    }
}

// Compute cross-entropy loss
double compute_loss(Network *net, unsigned char target_label) {
    return -log(net->output[target_label] + 1e-9);
}

// Apply dropout
double dropout(double x, double dropout_rate) {
    return ((double)rand() / RAND_MAX) > dropout_rate ? x : 0.0;
}

// Predict the label with the highest probability
int predict(Network *net) {
    int max_index = 0;
    for (int i = 1; i < net->output_size; ++i) {
        if (net->output[i] > net->output[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

// Save the network to a file
void save_network(const char *filename, Network *net) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for saving");
        exit(EXIT_FAILURE);
    }

    fwrite(net->weights1, sizeof(double), net->input_size * net->hidden_size1, file);
    fwrite(net->weights2, sizeof(double), net->hidden_size1 * net->hidden_size2, file);
    fwrite(net->weights3, sizeof(double), net->hidden_size2 * net->output_size, file);
    fwrite(net->biases1, sizeof(double), net->hidden_size1, file);
    fwrite(net->biases2, sizeof(double), net->hidden_size2, file);
    fwrite(net->biases3, sizeof(double), net->output_size, file);

    fclose(file);
    printf("Network saved to %s\n", filename);
}

// Load the network from a file
void load_network(const char *filename, Network *net) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for loading");
        exit(EXIT_FAILURE);
    }

    fread(net->weights1, sizeof(double), net->input_size * net->hidden_size1, file);
    fread(net->weights2, sizeof(double), net->hidden_size1 * net->hidden_size2, file);
    fread(net->weights3, sizeof(double), net->hidden_size2 * net->output_size, file);
    fread(net->biases1, sizeof(double), net->hidden_size1, file);
    fread(net->biases2, sizeof(double), net->hidden_size2, file);
    fread(net->biases3, sizeof(double), net->output_size, file);

    fclose(file);
    printf("Network loaded from %s\n", filename);
}
