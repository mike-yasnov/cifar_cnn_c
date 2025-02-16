#include "fc_layer.h"
#include <string.h>

void fc_forward(const FCLayer *fc, const float *input) {
    float *fc_input = (float *)fc->input;  // временное приведение типа для совместимости
    float *fc_output = (float *)fc->output;  // временное приведение типа для совместимости
    memcpy(fc_input, input, sizeof(float) * fc->input_size);
    for (int i = 0; i < fc->output_size; i++) {
        float sum = fc->biases[i];
        for (int j = 0; j < fc->input_size; j++) {
            sum += fc->weights[i * fc->input_size + j] * input[j];
        }
        fc_output[i] = sum;
    }
}

void fc_backward(FCLayer *fc, float *dout, float *dinput) {
    for (int i = 0; i < fc->output_size; i++) {
        fc->dbiases[i] = dout[i];
        for (int j = 0; j < fc->input_size; j++) {
            fc->dweights[i * fc->input_size + j] = dout[i] * fc->input[j];
        }
    }
    for (int j = 0; j < fc->input_size; j++) {
        float sum = 0.0f;
        for (int i = 0; i < fc->output_size; i++) {
            sum += fc->weights[i * fc->input_size + j] * dout[i];
        }
        dinput[j] = sum;
    }
}