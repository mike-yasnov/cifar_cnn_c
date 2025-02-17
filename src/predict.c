#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "conv_layer.h"
#include "fc_layer.h"
#include "maxpool_layer.h"
#include "activations.h"
#include "softmax.h"
#include "utils.h"
#include "dataset.h"

#define IMG_WIDTH    32
#define IMG_HEIGHT   32
#define IMG_CHANNELS 3
#define NUM_CLASSES  10

#define CONV1_KERNEL_SIZE 3
#define CONV1_OUT_CHANNELS 32
#define CONV1_OUT_HEIGHT (IMG_HEIGHT - CONV1_KERNEL_SIZE + 1)
#define CONV1_OUT_WIDTH  (IMG_WIDTH - CONV1_KERNEL_SIZE + 1)

#define CONV2_KERNEL_SIZE 3
#define CONV2_OUT_CHANNELS 64
#define CONV2_OUT_HEIGHT (CONV1_OUT_HEIGHT - CONV2_KERNEL_SIZE + 1)
#define CONV2_OUT_WIDTH  (CONV1_OUT_WIDTH - CONV2_KERNEL_SIZE + 1)

#define POOL_SIZE 2
#define POOL_OUT_HEIGHT (CONV2_OUT_HEIGHT / POOL_SIZE)
#define POOL_OUT_WIDTH  (CONV2_OUT_WIDTH / POOL_SIZE)

#define FC_INPUT_SIZE (CONV2_OUT_CHANNELS * POOL_OUT_HEIGHT * POOL_OUT_WIDTH)
#define FC_OUTPUT_SIZE NUM_CLASSES

// Функция для нормализации изображения
void normalize_image(float *image) {
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std[3] = {0.0f, 0.0f, 0.0f};
    int pixels_per_channel = IMG_HEIGHT * IMG_WIDTH;
    
    // Вычисляем среднее для каждого канала
    for (int c = 0; c < IMG_CHANNELS; c++) {
        for (int i = 0; i < pixels_per_channel; i++) {
            mean[c] += image[c * pixels_per_channel + i];
        }
        mean[c] /= pixels_per_channel;
    }
    
    // Вычисляем стандартное отклонение для каждого канала
    for (int c = 0; c < IMG_CHANNELS; c++) {
        for (int i = 0; i < pixels_per_channel; i++) {
            float diff = image[c * pixels_per_channel + i] - mean[c];
            std[c] += diff * diff;
        }
        std[c] = sqrt(std[c] / pixels_per_channel);
        if (std[c] < 1e-6f) std[c] = 1.0f;
    }
    
    // Нормализуем каждый канал
    for (int c = 0; c < IMG_CHANNELS; c++) {
        for (int i = 0; i < pixels_per_channel; i++) {
            image[c * pixels_per_channel + i] = 
                (image[c * pixels_per_channel + i] - mean[c]) / std[c];
        }
    }
}

int main(int argc, char *argv[]) {
    const char *data_dir = "cifar-10-batches-bin";
    
    Dataset test = load_cifar10_test(data_dir);
    printf("Загружено %d тестовых изображений.\n", test.num_samples);
    
    // Первый сверточный слой
    ConvLayer conv1;
    conv1.in_channels = IMG_CHANNELS;
    conv1.out_channels = CONV1_OUT_CHANNELS;
    conv1.kernel_size = CONV1_KERNEL_SIZE;
    conv1.in_height = IMG_HEIGHT;
    conv1.in_width = IMG_WIDTH;
    conv1.out_height = CONV1_OUT_HEIGHT;
    conv1.out_width = CONV1_OUT_WIDTH;
    int conv1_weight_size = conv1.out_channels * conv1.in_channels * conv1.kernel_size * conv1.kernel_size;
    conv1.weights = (float*)malloc(conv1_weight_size * sizeof(float));
    conv1.biases = (float*)malloc(conv1.out_channels * sizeof(float));
    conv1.input = (float*)malloc(conv1.in_channels * conv1.in_height * conv1.in_width * sizeof(float));
    conv1.output = (float*)malloc(conv1.out_channels * conv1.out_height * conv1.out_width * sizeof(float));
    
    // Второй сверточный слой
    ConvLayer conv2;
    conv2.in_channels = CONV1_OUT_CHANNELS;
    conv2.out_channels = CONV2_OUT_CHANNELS;
    conv2.kernel_size = CONV2_KERNEL_SIZE;
    conv2.in_height = CONV1_OUT_HEIGHT;
    conv2.in_width = CONV1_OUT_WIDTH;
    conv2.out_height = CONV2_OUT_HEIGHT;
    conv2.out_width = CONV2_OUT_WIDTH;
    int conv2_weight_size = conv2.out_channels * conv2.in_channels * conv2.kernel_size * conv2.kernel_size;
    conv2.weights = (float*)malloc(conv2_weight_size * sizeof(float));
    conv2.biases = (float*)malloc(conv2.out_channels * sizeof(float));
    conv2.input = (float*)malloc(conv2.in_channels * conv2.in_height * conv2.in_width * sizeof(float));
    conv2.output = (float*)malloc(conv2.out_channels * conv2.out_height * conv2.out_width * sizeof(float));
    
    MaxPoolLayer pool;
    pool.channels = conv2.out_channels;
    pool.in_height = conv2.out_height;
    pool.in_width = conv2.out_width;
    pool.pool_size = POOL_SIZE;
    pool.out_height = POOL_OUT_HEIGHT;
    pool.out_width = POOL_OUT_WIDTH;
    int pool_output_size = pool.channels * pool.out_height * pool.out_width;
    pool.input = (float*)malloc(pool.channels * pool.in_height * pool.in_width * sizeof(float));
    pool.output = (float*)malloc(pool_output_size * sizeof(float));
    pool.max_index = (int*)malloc(pool_output_size * sizeof(int));
    
    FCLayer fc;
    fc.input_size = FC_INPUT_SIZE;
    fc.output_size = FC_OUTPUT_SIZE;
    int fc_weight_size = fc.output_size * fc.input_size;
    fc.weights = (float*)malloc(fc_weight_size * sizeof(float));
    fc.biases = (float*)malloc(fc.output_size * sizeof(float));
    fc.input = (float*)malloc(fc.input_size * sizeof(float));
    fc.output = (float*)malloc(fc.output_size * sizeof(float));
    
    float *softmax_probs = (float*)malloc(fc.output_size * sizeof(float));
    int conv1_output_size = conv1.out_channels * conv1.out_height * conv1.out_width;
    float *relu1_out = (float*)malloc(conv1_output_size * sizeof(float));
    int conv2_output_size = conv2.out_channels * conv2.out_height * conv2.out_width;
    float *relu2_out = (float*)malloc(conv2_output_size * sizeof(float));

    // Загружаем сохраненные веса
    if (model_load_weights("weights/conv1_weights.bin", conv1.weights, conv1_weight_size) == 0 &&
        model_load_weights("weights/conv1_biases.bin", conv1.biases, conv1.out_channels) == 0 &&
        model_load_weights("weights/conv2_weights.bin", conv2.weights, conv2_weight_size) == 0 &&
        model_load_weights("weights/conv2_biases.bin", conv2.biases, conv2.out_channels) == 0 &&
        model_load_weights("weights/fc_weights.bin", fc.weights, fc_weight_size) == 0 &&
        model_load_weights("weights/fc_biases.bin", fc.biases, fc.output_size) == 0) {
        
        printf("Веса успешно загружены. Начинаем инференс...\n");

        int num_images = 50;
        if (argc > 1) {
            num_images = atoi(argv[1]);
            if (num_images <= 0 || num_images > test.num_samples) {
                num_images = 10;
            }
        }

        int total_correct = 0;
        for (int i = 0; i < num_images; i++) {
            float *image = test.images + i * IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH;
            int true_label = test.labels[i];
            
            // Нормализуем изображение перед прямым проходом
            normalize_image(image);
            
            // Прямой проход через сеть
            conv_forward(&conv1, image);
            relu_forward(conv1.output, relu1_out, conv1_output_size);
            conv_forward(&conv2, relu1_out);
            relu_forward(conv2.output, relu2_out, conv2_output_size);
            maxpool_forward(&pool, relu2_out);
            memcpy(fc.input, pool.output, sizeof(float) * fc.input_size);
            fc_forward(&fc, fc.input);
            softmax(fc.output, fc.output_size, 0, softmax_probs);
            
            // Находим класс с максимальной вероятностью
            int predicted_class = 0;
            float max_prob = softmax_probs[0];
            for (int j = 1; j < fc.output_size; j++) {
                if (softmax_probs[j] > max_prob) {
                    max_prob = softmax_probs[j];
                    predicted_class = j;
                }
            }
            
            // Сохраняем результат
            char filename[100];
            snprintf(filename, sizeof(filename), "predictions/prediction_%d.txt", i);
            model_save_prediction_result(filename, image, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,
                                      model_get_class_name(predicted_class));
            
            printf("Изображение %d:\n", i + 1);
            printf("Предсказание: %s (уверенность: %.2f%%)\n", 
                   model_get_class_name(predicted_class), max_prob * 100);
            printf("Истинный класс: %s\n\n", model_get_class_name(true_label));
            
            if (predicted_class == true_label) {
                total_correct++;
            }
        }
        
        printf("\nОбщая точность на %d изображениях: %.2f%%\n", 
               num_images, (100.0f * total_correct) / num_images);
    } else {
        printf("Ошибка: Не удалось загрузить веса модели\n");
    }
    
    // Освобождаем память
    free(test.images);
    free(test.labels);
    
    free(conv1.weights); free(conv1.biases);
    free(conv1.input); free(conv1.output);
    free(conv2.weights); free(conv2.biases);
    free(conv2.input); free(conv2.output);
    free(pool.input); free(pool.output); free(pool.max_index);
    free(fc.weights); free(fc.biases);
    free(fc.input); free(fc.output);
    free(softmax_probs);
    free(relu1_out);
    free(relu2_out);
    
    return 0;
} 