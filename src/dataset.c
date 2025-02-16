#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>

#define IMG_WIDTH    32
#define IMG_HEIGHT   32
#define IMG_CHANNELS 3
#define IMAGE_SIZE   (IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS) // 3072
#define RECORD_BYTES (1 + IMAGE_SIZE)                         // 3073

#define NUM_TRAIN_IMAGES 50000
#define NUM_TEST_IMAGES  10000

// Загрузка одного батча CIFAR-10 из бинарного файла
static void load_cifar10_batch(const char *file_path, float *images_out, int *labels_out, int num_images) {
    FILE *fp = fopen(file_path, "rb");
    if (!fp) {
        fprintf(stderr, "Не удалось открыть файл: %s\n", file_path);
        exit(1);
    }
    for (int i = 0; i < num_images; i++) {
        unsigned char label;
        if (fread(&label, 1, 1, fp) != 1) {
            fprintf(stderr, "Ошибка чтения метки из %s\n", file_path);
            exit(1);
        }
        labels_out[i] = (int)label;
        unsigned char buffer[IMAGE_SIZE];
        if (fread(buffer, 1, IMAGE_SIZE, fp) != IMAGE_SIZE) {
            fprintf(stderr, "Ошибка чтения изображения из %s\n", file_path);
            exit(1);
        }
        for (int j = 0; j < IMAGE_SIZE; j++) {
            images_out[i * IMAGE_SIZE + j] = (float)buffer[j] / 255.0f;
        }
    }
    fclose(fp);
}

Dataset load_cifar10_training(const char *data_dir) {
    Dataset ds;
    ds.num_samples = NUM_TRAIN_IMAGES;
    ds.images = (float *)malloc(NUM_TRAIN_IMAGES * IMAGE_SIZE * sizeof(float));
    ds.labels = (int *)malloc(NUM_TRAIN_IMAGES * sizeof(int));
    
    const char *batch_files[5] = {
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"
    };
    
    int offset = 0;
    for (int b = 0; b < 5; b++) {
        char filepath[256];
        snprintf(filepath, sizeof(filepath), "%s/%s", data_dir, batch_files[b]);
        load_cifar10_batch(filepath, ds.images + offset * IMAGE_SIZE, ds.labels + offset, 10000);
        offset += 10000;
    }
    
    return ds;
}

Dataset load_cifar10_test(const char *data_dir) {
    Dataset ds;
    ds.num_samples = NUM_TEST_IMAGES;
    ds.images = (float *)malloc(NUM_TEST_IMAGES * IMAGE_SIZE * sizeof(float));
    ds.labels = (int *)malloc(NUM_TEST_IMAGES * sizeof(int));
    
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "%s/%s", data_dir, "test_batch.bin");
    load_cifar10_batch(filepath, ds.images, ds.labels, NUM_TEST_IMAGES);
    
    return ds;
}