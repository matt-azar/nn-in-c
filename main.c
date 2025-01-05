#include "mnist_loader.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPOCHS 10
#define DROPOUT_RATE 0.0

int main() {
    double learning_rate = 1.6E-3;
    int load = 0;
    int correct = 0;

    puts("Load network from nn_weights.bin? [y/N]");
    char l = getchar();
    if (l == 'y')
        load = 1;

    time_t start = time(NULL);
    srand(time(NULL));

    const char *train_image_file = "./raw/train-images-idx3-ubyte";
    const char *train_label_file = "./raw/train-labels-idx1-ubyte";
    const char *test_image_file = "./raw/t10k-images-idx3-ubyte";
    const char *test_label_file = "./raw/t10k-labels-idx1-ubyte";

    int train_num_images, train_num_labels, test_num_images, test_num_labels;
    unsigned char **train_images = load_images(train_image_file, &train_num_images);
    unsigned char *train_labels = load_labels(train_label_file, &train_num_labels);
    unsigned char **test_images = load_images(test_image_file, &test_num_images);
    unsigned char *test_labels = load_labels(test_label_file, &test_num_labels);

    if (train_num_images != train_num_labels || test_num_images != test_num_labels) {
        fprintf(stderr, "Mismatch between number of images and labels.\n");
        exit(EXIT_FAILURE);
    }

    Network net;
    initialize_network(&net, IMAGE_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE);

    if (load) {
        load_network("nn_weights.bin", &net);
        goto L1;
    }

    printf("Training samples: %d\nTesting samples: %d\n\n", train_num_images, test_num_images);

    initialize_weights(net.weights1, IMAGE_SIZE * HIDDEN_SIZE1, IMAGE_SIZE);
    initialize_weights(net.weights2, HIDDEN_SIZE1 * HIDDEN_SIZE2, HIDDEN_SIZE1);
    initialize_weights(net.weights3, HIDDEN_SIZE2 * OUTPUT_SIZE, HIDDEN_SIZE2);
    initialize_biases(net.biases1, HIDDEN_SIZE1);
    initialize_biases(net.biases2, HIDDEN_SIZE2);
    initialize_biases(net.biases3, OUTPUT_SIZE);

    printf("Running %d epochs...\n", EPOCHS);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double epoch_loss = 0.0;

        for (int i = 0; i < train_num_images; ++i) {
            double input[IMAGE_SIZE];
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                input[j] = train_images[i][j] / 255.0;
            }

            forward_pass(&net, input, DROPOUT_RATE, 1);
            backpropagate(&net, input, train_labels[i], learning_rate);
            epoch_loss += compute_loss(&net, train_labels[i]);

            if ((i + 1) % 1000 == 0 || i == train_num_images - 1) {
                printf("Epoch %d: Sample %d / %d\r", epoch + 1, i + 1, train_num_images);
                fflush(stdout);
            }
        }

        printf("\n\t Loss: %.4f\n", epoch_loss / train_num_images);
        learning_rate *= 0.9;
        // learning_rate *= 1.0 - 1.0 / EPOCHS;
        printf("\t Learning rate: %lf\n", learning_rate);
    }

    save_network("nn_weights.bin", &net);

L1:
    for (int i = 0; i < test_num_images; ++i) {
        double input[IMAGE_SIZE];
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            input[j] = test_images[i][j] / 255.0;
        }
        forward_pass(&net, input, 0.0, 0);
        if (predict(&net) == test_labels[i])
            ++correct;
    }

    printf("\nTest Accuracy: %.2f%%\n", (double)correct / test_num_images * 100);

    for (int i = 0; i < train_num_images; ++i)
        free(train_images[i]);
    free(train_images);
    free(train_labels);
    for (int i = 0; i < test_num_images; ++i)
        free(test_images[i]);
    free(test_images);
    free(test_labels);

    free_network(&net);
    printf("Runtime: %ld seconds.\n", time(NULL) - start);
}
