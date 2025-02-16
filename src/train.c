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
#define FC_HIDDEN_SIZE 1024
#define FC_OUTPUT_SIZE NUM_CLASSES

int main(int argc, char *argv[]) {
    srand(time(NULL));

    const char *data_dir = "cifar-10-batches-bin";
    
    Dataset train = load_cifar10_training(data_dir);
    printf("Загружено %d тренировочных изображений.\n", train.num_samples);
    
    // Нормализация данных
    printf("Нормализация данных...\n");
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std[3] = {0.0f, 0.0f, 0.0f};
    
    // Вычисляем среднее и стандартное отклонение за один проход
    for (int i = 0; i < train.num_samples; i++) {
        if (i % 10000 == 0) {
            printf("Обработано %d изображений...\n", i);
        }
        for (int c = 0; c < IMG_CHANNELS; c++) {
            for (int h = 0; h < IMG_HEIGHT; h++) {
                for (int w = 0; w < IMG_WIDTH; w++) {
                    int idx = i * (IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH) + 
                             c * (IMG_HEIGHT * IMG_WIDTH) + 
                             h * IMG_WIDTH + w;
                    float pixel = train.images[idx];
                    mean[c] += pixel;
                    std[c] += pixel * pixel;
                }
            }
        }
    }
    
    int pixels_per_channel = train.num_samples * IMG_HEIGHT * IMG_WIDTH;
    for (int c = 0; c < IMG_CHANNELS; c++) {
        mean[c] /= pixels_per_channel;
        float variance = std[c] / pixels_per_channel - mean[c] * mean[c];
        std[c] = sqrt(variance + 1e-8f);
    }
    
    // Применяем нормализацию
    printf("Применяем нормализацию...\n");
    for (int i = 0; i < train.num_samples; i++) {
        if (i % 10000 == 0) {
            printf("Нормализовано %d изображений...\n", i);
        }
        for (int c = 0; c < IMG_CHANNELS; c++) {
            for (int h = 0; h < IMG_HEIGHT; h++) {
                for (int w = 0; w < IMG_WIDTH; w++) {
                    int idx = i * (IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH) + 
                             c * (IMG_HEIGHT * IMG_WIDTH) + 
                             h * IMG_WIDTH + w;
                    train.images[idx] = (train.images[idx] - mean[c]) / std[c];
                }
            }
        }
    }
    printf("Нормализация завершена.\n\n");
    
    // Первый сверточный слой
    ConvLayer conv1;
    conv1.in_channels = IMG_CHANNELS;
    conv1.out_channels = CONV1_OUT_CHANNELS;
    conv1.kernel_size = CONV1_KERNEL_SIZE;
    conv1.in_height = IMG_HEIGHT;
    conv1.in_width = IMG_WIDTH;
    conv1.out_height = CONV1_OUT_HEIGHT;
    conv1.out_width = CONV1_OUT_WIDTH;
    
    // Второй сверточный слой
    ConvLayer conv2;
    conv2.in_channels = CONV1_OUT_CHANNELS;
    conv2.out_channels = CONV2_OUT_CHANNELS;
    conv2.kernel_size = CONV2_KERNEL_SIZE;
    conv2.in_height = CONV1_OUT_HEIGHT;
    conv2.in_width = CONV1_OUT_WIDTH;
    conv2.out_height = CONV2_OUT_HEIGHT;
    conv2.out_width = CONV2_OUT_WIDTH;
    
    // Выделение памяти и инициализация для conv1
    int conv1_weight_size = conv1.out_channels * conv1.in_channels * conv1.kernel_size * conv1.kernel_size;
    conv1.weights = (float*)malloc(conv1_weight_size * sizeof(float));
    conv1.biases = (float*)malloc(conv1.out_channels * sizeof(float));
    conv1.dweights = (float*)malloc(conv1_weight_size * sizeof(float));
    conv1.dbiases = (float*)malloc(conv1.out_channels * sizeof(float));
    conv1.input = (float*)malloc(conv1.in_channels * conv1.in_height * conv1.in_width * sizeof(float));
    conv1.output = (float*)malloc(conv1.out_channels * conv1.out_height * conv1.out_width * sizeof(float));
    
    // Инициализация Xavier для conv1
    float conv1_weight_scale = sqrt(2.0f / (conv1.in_channels * conv1.kernel_size * conv1.kernel_size));
    for (int i = 0; i < conv1_weight_size; i++) {
        conv1.weights[i] = ((float)rand()/RAND_MAX * 2 - 1) * conv1_weight_scale;
    }
    for (int i = 0; i < conv1.out_channels; i++) {
        conv1.biases[i] = 0.0f;
    }
    
    // Выделение памяти и инициализация для conv2
    int conv2_weight_size = conv2.out_channels * conv2.in_channels * conv2.kernel_size * conv2.kernel_size;
    conv2.weights = (float*)malloc(conv2_weight_size * sizeof(float));
    conv2.biases = (float*)malloc(conv2.out_channels * sizeof(float));
    conv2.dweights = (float*)malloc(conv2_weight_size * sizeof(float));
    conv2.dbiases = (float*)malloc(conv2.out_channels * sizeof(float));
    conv2.input = (float*)malloc(conv2.in_channels * conv2.in_height * conv2.in_width * sizeof(float));
    conv2.output = (float*)malloc(conv2.out_channels * conv2.out_height * conv2.out_width * sizeof(float));
    
    // Инициализация Xavier для conv2
    float conv2_weight_scale = sqrt(2.0f / (conv2.in_channels * conv2.kernel_size * conv2.kernel_size));
    for (int i = 0; i < conv2_weight_size; i++) {
        conv2.weights[i] = ((float)rand()/RAND_MAX * 2 - 1) * conv2_weight_scale;
    }
    for (int i = 0; i < conv2.out_channels; i++) {
        conv2.biases[i] = 0.0f;
    }
    
    MaxPoolLayer pool;
    pool.channels = conv2.out_channels;
    pool.in_height = conv2.out_height;
    pool.in_width = conv2.out_width;
    pool.pool_size = POOL_SIZE;
    pool.out_height = POOL_OUT_HEIGHT;
    pool.out_width = POOL_OUT_WIDTH;
    int pool_input_size = pool.channels * pool.in_height * pool.in_width;
    int pool_output_size = pool.channels * pool.out_height * pool.out_width;
    pool.input = (float*)malloc(pool_input_size * sizeof(float));
    pool.output = (float*)malloc(pool_output_size * sizeof(float));
    pool.max_index = (int*)malloc(pool_output_size * sizeof(int));
    
    FCLayer fc;
    fc.input_size = FC_INPUT_SIZE;
    fc.output_size = FC_OUTPUT_SIZE;
    int fc_weight_size = fc.output_size * fc.input_size;
    fc.weights = (float*)malloc(fc_weight_size * sizeof(float));
    fc.biases = (float*)malloc(fc.output_size * sizeof(float));
    fc.dweights = (float*)malloc(fc_weight_size * sizeof(float));
    fc.dbiases = (float*)malloc(fc.output_size * sizeof(float));
    fc.input = (float*)malloc(fc.input_size * sizeof(float));
    fc.output = (float*)malloc(fc.output_size * sizeof(float));
    for (int i = 0; i < fc_weight_size; i++) {
        fc.weights[i] = (((float)rand()/RAND_MAX) - 0.5f) * 0.1f;
    }
    for (int i = 0; i < fc.output_size; i++) {
        fc.biases[i] = 0.0f;
    }
    
    float *conv_dout = (float*)malloc(conv2.out_channels * conv2.out_height * conv2.out_width * sizeof(float));
    float *conv_input_grad = (float*)malloc(conv2.in_channels * conv2.in_height * conv2.in_width * sizeof(float));
    float *pool_dout = (float*)malloc(pool_output_size * sizeof(float));
    float *fc_dinput = (float*)malloc(fc.input_size * sizeof(float));
    float *softmax_probs = (float*)malloc(fc.output_size * sizeof(float));
    float *fc_dout = (float*)malloc(fc.output_size * sizeof(float));
    
    // Буферы для первого сверточного слоя
    int conv1_output_size = conv1.out_channels * conv1.out_height * conv1.out_width;
    float *relu1_out = (float*)malloc(conv1_output_size * sizeof(float));
    float *relu1_dout = (float*)malloc(conv1_output_size * sizeof(float));
    
    // Буферы для второго сверточного слоя
    int conv2_output_size = conv2.out_channels * conv2.out_height * conv2.out_width;
    float *relu2_out = (float*)malloc(conv2_output_size * sizeof(float));
    float *relu2_dout = (float*)malloc(conv2_output_size * sizeof(float));
    
    int epochs = 100;
    float initial_learning_rate = 0.01f;
    float learning_rate = initial_learning_rate;
    int train_samples = 10000;
    
    float best_val_accuracy = 0.0f;
    int patience = 5;
    int epochs_without_improvement = 0;
    
    // Режим обучения
    for (int epoch = 0; epoch < epochs; epoch++) {
        if (epochs_without_improvement >= patience) {
            learning_rate *= 0.5f;
            epochs_without_improvement = 0;
            printf("Уменьшаем learning rate до %f\n", learning_rate);
        }
        
        float epoch_loss = 0.0f;
        int correct = 0;
        
        for (int i = 0; i < train_samples; i++) {
            int j = i + rand() % (train_samples - i);
            for (int k = 0; k < IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH; k++) {
                float temp = train.images[i * IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH + k];
                train.images[i * IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH + k] = 
                    train.images[j * IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH + k];
                train.images[j * IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH + k] = temp;
            }
            int temp_label = train.labels[i];
            train.labels[i] = train.labels[j];
            train.labels[j] = temp_label;
        }
        
        for (int n = 0; n < train_samples; n++) {
            float *image = train.images + n * IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH;
            int label = train.labels[n];
            
            conv_forward(&conv1, image);
            relu_forward(conv1.output, relu1_out, conv1_output_size);
            conv_forward(&conv2, relu1_out);
            relu_forward(conv2.output, relu2_out, conv2_output_size);
            maxpool_forward(&pool, relu2_out);
            memcpy(fc.input, pool.output, sizeof(float) * fc.input_size);
            fc_forward(&fc, fc.input);
            float loss = softmax(fc.output, fc.output_size, label, softmax_probs);
            epoch_loss += loss;
            
            int pred = 0;
            float max_prob = softmax_probs[0];
            for (int i = 1; i < fc.output_size; i++) {
                if (softmax_probs[i] > max_prob) {
                    max_prob = softmax_probs[i];
                    pred = i;
                }
            }
            if (pred == label)
                correct++;
            
            softmax_backward(softmax_probs, label, fc.output_size, fc_dout);
            fc_backward(&fc, fc_dout, fc_dinput);
            memcpy(pool_dout, fc_dinput, sizeof(float) * fc.input_size);
            maxpool_backward(&pool, pool_dout, relu2_dout);
            relu_backward(conv2.output, relu2_dout, conv_dout, conv2_output_size);
            conv_backward(&conv2, conv_dout, conv_input_grad);
            relu_backward(conv1.output, conv_input_grad, relu1_dout, conv1_output_size);
            conv_backward(&conv1, relu1_dout, conv_input_grad);
            
            model_update_parameters(conv1.weights, conv1.dweights, conv1_weight_size, learning_rate);
            model_update_parameters(conv1.biases, conv1.dbiases, conv1.out_channels, learning_rate);
            model_update_parameters(conv2.weights, conv2.dweights, conv2_weight_size, learning_rate);
            model_update_parameters(conv2.biases, conv2.dbiases, conv2.out_channels, learning_rate);
            model_update_parameters(fc.weights, fc.dweights, fc_weight_size, learning_rate);
            model_update_parameters(fc.biases, fc.dbiases, fc.output_size, learning_rate);
        }
        printf("Epoch %d: Loss = %f, Train Accuracy = %.2f%%\n",
               epoch + 1, epoch_loss / train_samples, (100.0 * correct) / train_samples);

        int val_samples = 1000;
        int val_correct = 0;
        for (int n = train_samples; n < train_samples + val_samples; n++) {
            float *image = train.images + n * IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH;
            int label = train.labels[n];
            
            conv_forward(&conv1, image);
            relu_forward(conv1.output, relu1_out, conv1_output_size);
            conv_forward(&conv2, relu1_out);
            relu_forward(conv2.output, relu2_out, conv2_output_size);
            maxpool_forward(&pool, relu2_out);
            memcpy(fc.input, pool.output, sizeof(float) * fc.input_size);
            fc_forward(&fc, fc.input);
            softmax(fc.output, fc.output_size, label, softmax_probs);
            
            int pred = 0;
            float max_prob = softmax_probs[0];
            for (int i = 1; i < fc.output_size; i++) {
                if (softmax_probs[i] > max_prob) {
                    max_prob = softmax_probs[i];
                    pred = i;
                }
            }
            if (pred == label)
                val_correct++;
        }
        printf("Validation Accuracy = %.2f%%\n", (100.0 * val_correct) / val_samples);
        
        float current_val_accuracy = (100.0 * val_correct) / val_samples;
        if (current_val_accuracy > best_val_accuracy) {
            best_val_accuracy = current_val_accuracy;
            epochs_without_improvement = 0;
            printf("Новая лучшая точность! Сохраняем веса...\n\n");
            
            if (model_save_weights("weights/conv1_weights.bin", conv1.weights, conv1_weight_size) == 0) {
                model_save_weights("weights/conv1_biases.bin", conv1.biases, conv1.out_channels);
            }
            if (model_save_weights("weights/conv2_weights.bin", conv2.weights, conv2_weight_size) == 0) {
                model_save_weights("weights/conv2_biases.bin", conv2.biases, conv2.out_channels);
            }
            if (model_save_weights("weights/fc_weights.bin", fc.weights, fc_weight_size) == 0) {
                model_save_weights("weights/fc_biases.bin", fc.biases, fc.output_size);
            }
        } else {
            epochs_without_improvement++;
            printf("Точность не улучшилась. Лучшая точность: %.2f%%. Эпох без улучшения: %d\n\n", 
                   best_val_accuracy, epochs_without_improvement);
        }
    }
    
    free(train.images);
    free(train.labels);
    
    free(conv1.weights); free(conv1.biases); free(conv1.dweights); free(conv1.dbiases);
    free(conv1.input); free(conv1.output);
    free(conv2.weights); free(conv2.biases); free(conv2.dweights); free(conv2.dbiases);
    free(conv2.input); free(conv2.output);
    free(pool.input); free(pool.output); free(pool.max_index);
    free(fc.weights); free(fc.biases); free(fc.dweights); free(fc.dbiases);
    free(fc.input); free(fc.output);
    free(conv_dout); free(conv_input_grad);
    free(pool_dout); free(fc_dinput);
    free(softmax_probs); free(fc_dout);
    free(relu1_out); free(relu1_dout);
    free(relu2_out); free(relu2_dout);
    
    return 0;
} 