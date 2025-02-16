#include "maxpool_layer.h"
#include <string.h>
#include <math.h>
#include <float.h>

void maxpool_forward(const MaxPoolLayer *pool, const float *input) {
    int total_in = pool->channels * pool->in_height * pool->in_width;
    float *pool_input = (float *)pool->input;  // временное приведение типа для совместимости
    float *pool_output = (float *)pool->output;  // временное приведение типа для совместимости
    int *pool_max_index = (int *)pool->max_index;  // временное приведение типа для совместимости
    memcpy(pool_input, input, sizeof(float) * total_in);
    
    for (int c = 0; c < pool->channels; c++) {
        for (int oh = 0; oh < pool->out_height; oh++) {
            for (int ow = 0; ow < pool->out_width; ow++) {
                int h_start = oh * pool->pool_size;
                int w_start = ow * pool->pool_size;
                float max_val = -INFINITY;
                int max_idx = -1;
                for (int ph = 0; ph < pool->pool_size; ph++) {
                    for (int pw = 0; pw < pool->pool_size; pw++) {
                        int h = h_start + ph;
                        int w = w_start + pw;
                        int idx = c * (pool->in_height * pool->in_width) + h * pool->in_width + w;
                        if (input[idx] > max_val) {
                            max_val = input[idx];
                            max_idx = idx;
                        }
                    }
                }
                int out_idx = c * (pool->out_height * pool->out_width) + oh * pool->out_width + ow;
                pool_output[out_idx] = max_val;
                pool_max_index[out_idx] = max_idx;
            }
        }
    }
}

void maxpool_backward(MaxPoolLayer *pool, float *dout, float *dinput) {
    int total_in = pool->channels * pool->in_height * pool->in_width;
    for (int i = 0; i < total_in; i++)
        dinput[i] = 0.0f;
    
    for (int c = 0; c < pool->channels; c++) {
        for (int oh = 0; oh < pool->out_height; oh++) {
            for (int ow = 0; ow < pool->out_width; ow++) {
                int out_idx = c * (pool->out_height * pool->out_width) + oh * pool->out_width + ow;
                int max_idx = pool->max_index[out_idx];
                dinput[max_idx] += dout[out_idx];
            }
        }
    }
}