#ifndef UTILS_H
#define UTILS_H

#include "conv_layer.h"
#include "fc_layer.h"
#include "maxpool_layer.h"

// Существующие функции
void update_parameters(float *params, float *grads, int size, float learning_rate);
int save_weights(const char *filename, float *weights, int size);
int load_weights(const char *filename, float *weights, int size);

// Новые функции для инференса и работы с изображениями
const char* get_class_name(int class_id);
void save_prediction_result(const char *filename, float *image, int width, int height, int channels, const char *predicted_class);
int model_inference(float *image, ConvLayer *conv, MaxPoolLayer *pool, FCLayer *fc,
                   float *relu_out, float *softmax_probs);

#endif // UTILS_H