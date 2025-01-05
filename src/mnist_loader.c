#include "mnist_loader.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Read a 4-byte integer from a file in big-endian format
int read_int(FILE *file) {
    unsigned char bytes[4];
    size_t bytes_read = fread(bytes, sizeof(unsigned char), 4, file);

    if (bytes_read != 4) {
        fprintf(stderr, "Error: Failed to read 4 bytes from file.\n");
        exit(EXIT_FAILURE);
    }

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
        fclose(file);
        exit(EXIT_FAILURE);
    }

    unsigned char **images = malloc(*num_images * sizeof(unsigned char *));
    if (!images) {
        fprintf(stderr, "Failed to allocate memory for images.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < *num_images; ++i) {
        images[i] = malloc(rows * cols * sizeof(unsigned char));
        if (!images[i]) {
            fprintf(stderr, "Failed to allocate memory for image %d.\n", i);

            for (int j = 0; j < i; ++j) {
                free(images[j]);
            }
            free(images);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        size_t items_read = fread(images[i], sizeof(unsigned char), rows * cols, file);
        if (items_read != (size_t)(rows * cols)) {
            fprintf(stderr, "Failed to read image %d from file.\n", i);

            for (int j = 0; j <= i; ++j) {
                free(images[j]);
            }
            free(images);
            fclose(file);
            exit(EXIT_FAILURE);
        }
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
        fclose(file);
        exit(EXIT_FAILURE);
    }

    unsigned char *labels = malloc(*num_labels * sizeof(unsigned char));
    if (!labels) {
        fprintf(stderr, "Failed to allocate memory for labels.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    size_t items_read = fread(labels, sizeof(unsigned char), *num_labels, file);
    if (items_read != (size_t)(*num_labels)) {
        fprintf(stderr, "Failed to read labels from file: %s\n", filename);
        free(labels);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fclose(file);
    return labels;
}

// Display an ASCII MNIST image in the terminal
void display_image(unsigned char *image) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            printf("%c", image[i * 28 + j] > 128 ? '#' : '.');
        }
        printf("\n");
    }
}
