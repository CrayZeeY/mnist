#include <stdio.h>
#include <stdlib.h>
#include "data.h"
#include "network.h"

/* Reads 4 bytes and reverses them — MNIST uses big-endian */
static int read_int(FILE *f) {
    unsigned char b[4];
    fread(b, 1, 4, f);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

Dataset load_mnist(const char *image_file, const char *label_file) {
    Dataset d;
    d.count = 0; d.images = NULL; d.labels = NULL;

    FILE *img_f = fopen(image_file, "rb");
    FILE *lbl_f = fopen(label_file, "rb");
    if (!img_f || !lbl_f) {
        printf("Error: cannot open MNIST files\n");
        return d;
    }

    int img_magic = read_int(img_f);
    int lbl_magic = read_int(lbl_f);
    if (img_magic != 2051 || lbl_magic != 2049) {
        printf("Error: invalid MNIST format — are files unzipped?\n");
        return d;
    }

    int count = read_int(img_f);
    read_int(lbl_f);
    int rows = read_int(img_f);
    int cols = read_int(img_f);
    int pixels = rows * cols;

    d.count  = count;
    d.images = (double **)malloc(count * sizeof(double *));
    d.labels = (int *)malloc(count * sizeof(int));

    printf("Loading %d samples (%dx%d)...\n", count, rows, cols);

    for (int i = 0; i < count; i++) {
        d.images[i] = (double *)malloc(pixels * sizeof(double));

        for (int p = 0; p < pixels; p++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, img_f);
            d.images[i][p] = pixel / 255.0;   /* normalize to 0.0-1.0 */
        }

        unsigned char label;
        fread(&label, 1, 1, lbl_f);
        d.labels[i] = (int)label;
    }

    fclose(img_f); fclose(lbl_f);
    printf("Done.\n");
    return d;
}

void free_dataset(Dataset *d) {
    for (int i = 0; i < d->count; i++) free(d->images[i]);
    free(d->images);
    free(d->labels);
}

void print_image(double *image) {
    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            double p = image[row * 28 + col];
            if      (p > 0.7) printf("#");
            else if (p > 0.3) printf("+");
            else if (p > 0.0) printf(".");
            else              printf(" ");
        }
        printf("\n");
    }
}