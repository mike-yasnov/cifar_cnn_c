#ifndef UTILS_H
#define UTILS_H

#include "conv_layer.h"
#include "fc_layer.h"
#include "maxpool_layer.h"

// Функции для работы с параметрами модели
void model_update_parameters(float *params, float *grads, int size, float learning_rate);
int model_save_weights(const char *filename, const float *weights, int size);
int model_load_weights(const char *filename, float *weights, int size);
void model_save_prediction_result(const char *filename, const float *image, int width, int height, int channels, const char *prediction);
const char* model_get_class_name(int class_id);

// Функции для инференса и визуализации
void model_save_prediction(const char *filename, const float *image, 
                         int width, int height, int channels,
                         const char *predicted_class);
int model_inference(const float *image, const ConvLayer *conv, const MaxPoolLayer *pool,
                   const FCLayer *fc, float *relu_out, float *softmax_probs);

#endif // UTILS_H