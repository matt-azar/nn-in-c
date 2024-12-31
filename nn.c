#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/// @brief Initialize the weights for each neuron in the hidden layers using uniform Xavier initialization.
/// @param weights 
/// @param size 
/// @param inputs 
void initialize_weights(double *weights, int size, int inputs) {
    double limit = sqrt(6.0 / inputs);
    for (int i = 0; i < size; ++i) {
        weights[i] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
    }
}

/// @brief Initialize biases with a uniform distribution in [0, 0.01].
/// @param biases 
/// @param size 
void initialize_biases(double *biases, int size) {
    for (int i = 0; i < size; ++i) {
        biases[i] = ((double)rand() / RAND_MAX) / 100;
    }
}

/// @brief ReLU activation function.
/// @param x 
/// @return `ReLU(x) = max(0, x).`
double relu(double x) {
    return x > 0 ? x : 0;
}

/// @brief Derivative of ReLU activation function.
/// @param x 
/// @return (d/dx)ReLU(x) = max(0, 1).
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

/// @brief Softmax activation function.
/// @param output 
/// @param size 
void softmax(double *output, int size) {
    double max = output[0], sum = 0.0;
    for (int i = 1; i < size; ++i) {
        if (output[i] > max) max = output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i] = exp(output[i] - max);  // Subtract max for numerical stability
        sum += output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

/// @brief Performs a forward pass through a three-layer neural network.
///
/// This function computes the outputs of a feedforward neural network with three layers:
/// An input layer,
/// Two hidden layers with ReLU activation and optional dropout,
/// An output layer with softmax activation.
///
/// @param input Pointer to the input data array of size IMAGE_SIZE.
/// @param weights1 Pointer to the weight matrix for the first hidden layer (HIDDEN_SIZE1 x IMAGE_SIZE).
/// @param weights2 Pointer to the weight matrix for the second hidden layer (HIDDEN_SIZE2 x HIDDEN_SIZE1).
/// @param weights3 Pointer to the weight matrix for the output layer (OUTPUT_SIZE x HIDDEN_SIZE2).
/// @param biases1 Pointer to the bias vector for the first hidden layer of size HIDDEN_SIZE1.
/// @param biases2 Pointer to the bias vector for the second hidden layer of size HIDDEN_SIZE2.
/// @param biases3 Pointer to the bias vector for the output layer of size OUTPUT_SIZE.
/// @param hidden1 Pointer to the array that stores the activations of the first hidden layer (size HIDDEN_SIZE1).
/// @param hidden2 Pointer to the array that stores the activations of the second hidden layer (size HIDDEN_SIZE2).
/// @param output Pointer to the array that stores the output of the network (size OUTPUT_SIZE).
/// @param dropout_rate The dropout probability for regularization (0.0 to 1.0). Ignored if is_training is 0.
/// @param is_training Indicator of training mode (1 for training, 0 for inference). 
///                    When 1, dropout is applied to hidden layer activations.
///
/// The forward pass involves:
/// - Linear transformations (matrix multiplications with weights and addition of biases)
/// - ReLU activation for the hidden layers
/// - Dropout applied during training to prevent overfitting
/// - Softmax activation for the output layer to produce a probability distribution
void forward_pass(double *input, double *weights1, double *weights2, double *weights3,
                  double *biases1, double *biases2, double *biases3,
                  double *hidden1, double *hidden2, double *output, double dropout_rate, int is_training) {
    // First hidden layer
    for (int i = 0; i < HIDDEN_SIZE1; ++i) {
        hidden1[i] = biases1[i];  // Add bias
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            hidden1[i] += input[j] * weights1[i * IMAGE_SIZE + j];
        }
        hidden1[i] = relu(hidden1[i]);
        if (is_training) {
            hidden1[i] = dropout(hidden1[i], dropout_rate);
        }
    }

    // Second hidden layer
    for (int i = 0; i < HIDDEN_SIZE2; ++i) {
        hidden2[i] = biases2[i];  // Add bias
        for (int j = 0; j < HIDDEN_SIZE1; ++j) {
            hidden2[i] += hidden1[j] * weights2[i * HIDDEN_SIZE1 + j];
        }
        hidden2[i] = relu(hidden2[i]);
        if (is_training) {
            hidden2[i] = dropout(hidden2[i], dropout_rate);
        }
    }

    // Output layer
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = biases3[i];  // Add bias
        for (int j = 0; j < HIDDEN_SIZE2; ++j) {
            output[i] += hidden2[j] * weights3[i * HIDDEN_SIZE2 + j];
        }
    }

    softmax(output, OUTPUT_SIZE);
}

/// @brief Performs backpropagation for a three-layer neural network.
///
/// This function calculates the gradients of the weights and biases using the backpropagation
/// algorithm and updates them using stochastic gradient descent (SGD).
///
/// @param input Pointer to the input data array of size IMAGE_SIZE.
/// @param weights1 Pointer to the weight matrix for the first hidden layer (HIDDEN_SIZE1 x IMAGE_SIZE).
/// @param weights2 Pointer to the weight matrix for the second hidden layer (HIDDEN_SIZE2 x HIDDEN_SIZE1).
/// @param weights3 Pointer to the weight matrix for the output layer (OUTPUT_SIZE x HIDDEN_SIZE2).
/// @param biases1 Pointer to the bias vector for the first hidden layer of size HIDDEN_SIZE1.
/// @param biases2 Pointer to the bias vector for the second hidden layer of size HIDDEN_SIZE2.
/// @param biases3 Pointer to the bias vector for the output layer of size OUTPUT_SIZE.
/// @param hidden1 Pointer to the array that stores the activations of the first hidden layer (size HIDDEN_SIZE1).
/// @param hidden2 Pointer to the array that stores the activations of the second hidden layer (size HIDDEN_SIZE2).
/// @param output Pointer to the array that stores the output of the network (size OUTPUT_SIZE).
/// @param target_label The index of the correct class (0 to OUTPUT_SIZE-1).
///                     This is used to compute the target one-hot vector.
/// @param learning_rate The learning rate for gradient descent.
///
/// The backpropagation process includes:
/// - Computing the error and delta for the output layer based on cross-entropy loss.
/// - Propagating the error backward to compute deltas for the hidden layers.
/// - Updating weights and biases for each layer based on the deltas and activations of the previous layer.
///
/// Detailed steps:
/// 1. Compute output layer error and delta:
///    - Error: \f$ \text{error}_i = \text{target}_i - \text{output}_i \f$
///    - Delta: \f$ \delta_i = \text{error}_i \f$
/// 2. Compute hidden layer 2 error and delta:
///    - Error: \f$ \text{error}_j = \sum_{i} \delta_i \cdot \text{weights3}[i][j] \f$
///    - Delta: \f$ \delta_j = \text{error}_j \cdot \text{ReLU}'(\text{hidden2}_j) \f$
/// 3. Compute hidden layer 1 error and delta:
///    - Error: \f$ \text{error}_k = \sum_{j} \delta_j \cdot \text{weights2}[j][k] \f$
///    - Delta: \f$ \delta_k = \text{error}_k \cdot \text{ReLU}'(\text{hidden1}_k) \f$
/// 4. Update weights and biases for each layer:
///    - \f$ \text{weights3}[i][j] += \eta \cdot \delta_i \cdot \text{hidden2}_j \f$
///    - \f$ \text{weights2}[j][k] += \eta \cdot \delta_j \cdot \text{hidden1}_k \f$
///    - \f$ \text{weights1}[k][m] += \eta \cdot \delta_k \cdot \text{input}_m \f$
///
/// The target one-hot vector is computed using the provided target label.
void backpropagate(double *input, double *weights1, double *weights2, double *weights3,
                   double *biases1, double *biases2, double *biases3,
                   double *hidden1, double *hidden2, double *output, 
                   unsigned char target_label, double learning_rate) {
    double target[OUTPUT_SIZE] = {0};
    target[target_label] = 1.0;

    double output_error[OUTPUT_SIZE], output_delta[OUTPUT_SIZE];
    double hidden2_error[HIDDEN_SIZE2], hidden2_delta[HIDDEN_SIZE2];
    double hidden1_error[HIDDEN_SIZE1], hidden1_delta[HIDDEN_SIZE1];

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output_error[i] = target[i] - output[i];
        output_delta[i] = output_error[i];  // Cross-entropy loss gradient
        biases3[i] += learning_rate * output_delta[i];  // Update bias
    }

    for (int i = 0; i < HIDDEN_SIZE2; ++i) {
        hidden2_error[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            hidden2_error[i] += output_delta[j] * weights3[j * HIDDEN_SIZE2 + i];
        }
        hidden2_delta[i] = hidden2_error[i] * relu_derivative(hidden2[i]);
        biases2[i] += learning_rate * hidden2_delta[i];  // Update bias
    }

    for (int i = 0; i < HIDDEN_SIZE1; ++i) {
        hidden1_error[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE2; ++j) {
            hidden1_error[i] += hidden2_delta[j] * weights2[j * HIDDEN_SIZE1 + i];
        }
        hidden1_delta[i] = hidden1_error[i] * relu_derivative(hidden1[i]);
        biases1[i] += learning_rate * hidden1_delta[i];  // Update bias
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_SIZE2; ++j) {
            weights3[i * HIDDEN_SIZE2 + j] += learning_rate * output_delta[i] * hidden2[j];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE2; ++i) {
        for (int j = 0; j < HIDDEN_SIZE1; ++j) {
            weights2[i * HIDDEN_SIZE1 + j] += learning_rate * hidden2_delta[i] * hidden1[j];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE1; ++i) {
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            weights1[i * IMAGE_SIZE + j] += learning_rate * hidden1_delta[i] * input[j];
        }
    }
}

/**
 * @brief 
 * 
 * @param output 
 * @param target_label 
 * @return 
 */
double compute_loss(double *output, unsigned char target_label) {
    return -log(output[target_label] + 1e-9);  // Add small constant to avoid log(0)
}

double dropout(double x, double dropout_rate) {
    return ((double)rand() / RAND_MAX) > dropout_rate ? x : 0.0;
}

int predict(double *output) {
    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; ++i) {
        if (output[i] > output[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

void save_network(const char *filename, double *weights1, double *weights2, double *weights3,
                  double *biases1, double *biases2, double *biases3) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for saving");
        exit(EXIT_FAILURE);
    }

    fwrite(weights1, sizeof(double), IMAGE_SIZE * HIDDEN_SIZE1, file);
    fwrite(weights2, sizeof(double), HIDDEN_SIZE1 * HIDDEN_SIZE2, file);
    fwrite(weights3, sizeof(double), HIDDEN_SIZE2 * OUTPUT_SIZE, file);
    fwrite(biases1, sizeof(double), HIDDEN_SIZE1, file);
    fwrite(biases2, sizeof(double), HIDDEN_SIZE2, file);
    fwrite(biases3, sizeof(double), OUTPUT_SIZE, file);

    fclose(file);
    printf("Network saved to %s\n", filename);
}

void load_network(const char *filename, double *weights1, double *weights2, double *weights3,
                  double *biases1, double *biases2, double *biases3) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for loading");
        exit(EXIT_FAILURE);
    }

    fread(weights1, sizeof(double), IMAGE_SIZE * HIDDEN_SIZE1, file);
    fread(weights2, sizeof(double), HIDDEN_SIZE1 * HIDDEN_SIZE2, file);
    fread(weights3, sizeof(double), HIDDEN_SIZE2 * OUTPUT_SIZE, file);
    fread(biases1, sizeof(double), HIDDEN_SIZE1, file);
    fread(biases2, sizeof(double), HIDDEN_SIZE2, file);
    fread(biases3, sizeof(double), OUTPUT_SIZE, file);

    fclose(file);
    printf("Network loaded from %s\n", filename);
}
