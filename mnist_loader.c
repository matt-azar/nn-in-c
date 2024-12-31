#include "mnist_loader.h"
#include <stdlib.h>

// Read a 4-byte integer from a file in big-endian format
int read_int(FILE *file) {
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, file);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Load MNIST images
unsigned char **load_images(const char *filename, int *num_images) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    int magic_number = read_int(file);
    *num_images = read_int(file);
    int rows = read_int(file);
    int cols = read_int(file);

    if (magic_number != 0x00000803 || rows != 28 || cols != 28) {
        fprintf(stderr, "Invalid MNIST image file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    unsigned char **images = malloc(*num_images * sizeof(unsigned char *));
    for (int i = 0; i < *num_images; ++i) {
        images[i] = malloc(rows * cols * sizeof(unsigned char));
        fread(images[i], sizeof(unsigned char), rows * cols, file);
    }

    fclose(file);
    return images;
}

// Load MNIST labels
unsigned char *load_labels(const char *filename, int *num_labels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    int magic_number = read_int(file);
    *num_labels = read_int(file);

    if (magic_number != 0x00000801) {
        fprintf(stderr, "Invalid MNIST label file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    unsigned char *labels = malloc(*num_labels * sizeof(unsigned char));
    fread(labels, sizeof(unsigned char), *num_labels, file);

    fclose(file);
    return labels;
}

// Display an MNIST image in the terminal
void display_image(unsigned char *image) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            printf("%c", image[i * 28 + j] > 128 ? '#' : '.');
        }
        printf("\n");
    }
}
