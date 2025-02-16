#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

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

#define CONV_KERNEL_SIZE 3
#define CONV_OUT_CHANNELS 16
#define CONV_OUT_HEIGHT (IMG_HEIGHT - CONV_KERNEL_SIZE + 1)
#define CONV_OUT_WIDTH  (IMG_WIDTH - CONV_KERNEL_SIZE + 1)

#define POOL_SIZE 2
#define POOL_OUT_HEIGHT (CONV_OUT_HEIGHT / POOL_SIZE)
#define POOL_OUT_WIDTH  (CONV_OUT_WIDTH / POOL_SIZE)

#define FC_INPUT_SIZE (CONV_OUT_CHANNELS * POOL_OUT_HEIGHT * POOL_OUT_WIDTH)
#define FC_OUTPUT_SIZE NUM_CLASSES

int main(int argc, char *argv[]) {
    const char *data_dir = "cifar-10-batches-bin";
    
    // Загружаем тестовый набор
    Dataset test = load_cifar10_test(data_dir);
    printf("Загружено %d тестовых изображений.\n", test.num_samples);
    
    ConvLayer conv;
    conv.in_channels = IMG_CHANNELS;
    conv.out_channels = CONV_OUT_CHANNELS;
    conv.kernel_size = CONV_KERNEL_SIZE;
    conv.in_height = IMG_HEIGHT;
    conv.in_width = IMG_WIDTH;
    conv.out_height = CONV_OUT_HEIGHT;
    conv.out_width = CONV_OUT_WIDTH;
    int conv_weight_size = conv.out_channels * conv.in_channels * conv.kernel_size * conv.kernel_size;
    conv.weights = (float*)malloc(conv_weight_size * sizeof(float));
    conv.biases = (float*)malloc(conv.out_channels * sizeof(float));
    conv.input = (float*)malloc(conv.in_channels * conv.in_height * conv.in_width * sizeof(float));
    conv.output = (float*)malloc(conv.out_channels * conv.out_height * conv.out_width * sizeof(float));
    
    MaxPoolLayer pool;
    pool.channels = conv.out_channels;
    pool.in_height = conv.out_height;
    pool.in_width = conv.out_width;
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
    int conv_output_size = conv.out_channels * conv.out_height * conv.out_width;
    float *relu_out = (float*)malloc(conv_output_size * sizeof(float));

    // Загружаем сохраненные веса
    if (model_load_weights("weights/conv_weights.bin", conv.weights, conv_weight_size) == 0 &&
        model_load_weights("weights/conv_biases.bin", conv.biases, conv.out_channels) == 0 &&
        model_load_weights("weights/fc_weights.bin", fc.weights, fc_weight_size) == 0 &&
        model_load_weights("weights/fc_biases.bin", fc.biases, fc.output_size) == 0) {
        
        printf("Веса успешно загружены. Начинаем инференс...\n");

        // Определяем количество изображений для предсказания
        int num_images = 10;  // По умолчанию 10 изображений
        if (argc > 1) {
            num_images = atoi(argv[1]);
            if (num_images <= 0 || num_images > test.num_samples) {
                num_images = 10;
            }
        }

        // Проводим инференс
        int total_correct = 0;
        for (int i = 0; i < num_images; i++) {
            float *image = test.images + i * IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH;
            int true_label = test.labels[i];
            
            int predicted_class = model_inference(image, &conv, &pool, &fc, relu_out, softmax_probs);
            
            // Сохраняем результат в файл
            char filename[100];
            snprintf(filename, sizeof(filename), "predictions/prediction_%d.txt", i);
            model_save_prediction_result(filename, image, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,
                                model_get_class_name(predicted_class));
            
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
    
    free(conv.weights); free(conv.biases);
    free(conv.input); free(conv.output);
    free(pool.input); free(pool.output); free(pool.max_index);
    free(fc.weights); free(fc.biases);
    free(fc.input); free(fc.output);
    free(softmax_probs);
    free(relu_out);
    
    return 0;
} 