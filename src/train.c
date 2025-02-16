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
    srand(time(NULL));

    const char *data_dir = "cifar-10-batches-bin";
    
    Dataset train = load_cifar10_training(data_dir);
    printf("Загружено %d тренировочных изображений.\n", train.num_samples);
    
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
    conv.dweights = (float*)malloc(conv_weight_size * sizeof(float));
    conv.dbiases = (float*)malloc(conv.out_channels * sizeof(float));
    conv.input = (float*)malloc(conv.in_channels * conv.in_height * conv.in_width * sizeof(float));
    conv.output = (float*)malloc(conv.out_channels * conv.out_height * conv.out_width * sizeof(float));
    for (int i = 0; i < conv_weight_size; i++) {
        conv.weights[i] = (((float)rand()/RAND_MAX) - 0.5f) * 0.1f;
    }
    for (int i = 0; i < conv.out_channels; i++) {
        conv.biases[i] = 0.0f;
    }
    
    MaxPoolLayer pool;
    pool.channels = conv.out_channels;
    pool.in_height = conv.out_height;
    pool.in_width = conv.out_width;
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
    
    float *conv_dout = (float*)malloc(conv.out_channels * conv.out_height * conv.out_width * sizeof(float));
    float *conv_input_grad = (float*)malloc(conv.in_channels * conv.in_height * conv.in_width * sizeof(float));
    float *pool_dout = (float*)malloc(pool_output_size * sizeof(float));
    float *fc_dinput = (float*)malloc(fc.input_size * sizeof(float));
    float *softmax_probs = (float*)malloc(fc.output_size * sizeof(float));
    float *fc_dout = (float*)malloc(fc.output_size * sizeof(float));
    
    int conv_output_size = conv.out_channels * conv.out_height * conv.out_width;
    float *relu_out = (float*)malloc(conv_output_size * sizeof(float));
    float *relu_dout = (float*)malloc(conv_output_size * sizeof(float));
    
    int epochs = 20;
    float learning_rate = 0.01f;
    int train_samples = 1000;

    // Режим обучения
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        int correct = 0;
        for (int n = 0; n < train_samples; n++) {
            float *image = train.images + n * IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH;
            int label = train.labels[n];
            
            conv_forward(&conv, image);
            relu_forward(conv.output, relu_out, conv_output_size);
            maxpool_forward(&pool, relu_out);
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
            maxpool_backward(&pool, pool_dout, relu_dout);
            relu_backward(conv.output, relu_dout, conv_dout, conv_output_size);
            conv_backward(&conv, conv_dout, conv_input_grad);
            
            update_parameters(conv.weights, conv.dweights, conv_weight_size, learning_rate);
            update_parameters(conv.biases, conv.dbiases, conv.out_channels, learning_rate);
            update_parameters(fc.weights, fc.dweights, fc_weight_size, learning_rate);
            update_parameters(fc.biases, fc.dbiases, fc.output_size, learning_rate);
        }
        printf("Epoch %d: Loss = %f, Accuracy = %.2f%%\n",
               epoch + 1, epoch_loss / train_samples, (100.0 * correct) / train_samples);

        // Сохраняем веса сверточного слоя
        if (save_weights("weights/conv_weights.bin", conv.weights, conv_weight_size) == 0) {
            save_weights("weights/conv_biases.bin", conv.biases, conv.out_channels);
        }

        // Сохраняем веса полносвязного слоя
        if (save_weights("weights/fc_weights.bin", fc.weights, fc_weight_size) == 0) {
            save_weights("weights/fc_biases.bin", fc.biases, fc.output_size);
        }
    }
    
    free(train.images);
    free(train.labels);
    
    free(conv.weights); free(conv.biases); free(conv.dweights); free(conv.dbiases);
    free(conv.input); free(conv.output);
    free(pool.input); free(pool.output); free(pool.max_index);
    free(fc.weights); free(fc.biases); free(fc.dweights); free(fc.dbiases);
    free(fc.input); free(fc.output);
    free(conv_dout); free(conv_input_grad);
    free(pool_dout); free(fc_dinput);
    free(softmax_probs); free(fc_dout);
    free(relu_out); free(relu_dout);
    
    return 0;
} 