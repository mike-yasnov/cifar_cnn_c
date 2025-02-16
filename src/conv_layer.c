#include "conv_layer.h"
#include <string.h>
#include <math.h>

void conv_forward(ConvLayer *layer, float *input) {
    int in_size = layer->in_channels * layer->in_height * layer->in_width;
    memcpy(layer->input, input, sizeof(float) * in_size);
    
    for (int oc = 0; oc < layer->out_channels; oc++) {
        for (int oh = 0; oh < layer->out_height; oh++) {
            for (int ow = 0; ow < layer->out_width; ow++) {
                float sum = layer->biases[oc];
                for (int ic = 0; ic < layer->in_channels; ic++) {
                    for (int kh = 0; kh < layer->kernel_size; kh++) {
                        for (int kw = 0; kw < layer->kernel_size; kw++) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            int input_index = ic * (layer->in_height * layer->in_width) + ih * layer->in_width + iw;
                            int weight_index = oc * (layer->in_channels * layer->kernel_size * layer->kernel_size)
                                               + ic * (layer->kernel_size * layer->kernel_size)
                                               + kh * layer->kernel_size + kw;
                            sum += input[input_index] * layer->weights[weight_index];
                        }
                    }
                }
                int out_index = oc * (layer->out_height * layer->out_width) + oh * layer->out_width + ow;
                layer->output[out_index] = sum;
            }
        }
    }
}

void conv_backward(ConvLayer *layer, float *dout, float *dinput) {
    int in_size = layer->in_channels * layer->in_height * layer->in_width;
    int weight_size = layer->out_channels * layer->in_channels * layer->kernel_size * layer->kernel_size;
    
    // Обнуляем dinput, dweights и dbiases
    for (int i = 0; i < in_size; i++)
        dinput[i] = 0.0f;
    for (int i = 0; i < weight_size; i++)
        layer->dweights[i] = 0.0f;
    for (int oc = 0; oc < layer->out_channels; oc++)
        layer->dbiases[oc] = 0.0f;
    
    for (int oc = 0; oc < layer->out_channels; oc++) {
        for (int oh = 0; oh < layer->out_height; oh++) {
            for (int ow = 0; ow < layer->out_width; ow++) {
                int out_index = oc * (layer->out_height * layer->out_width) + oh * layer->out_width + ow;
                float grad = dout[out_index];
                layer->dbiases[oc] += grad;
                for (int ic = 0; ic < layer->in_channels; ic++) {
                    for (int kh = 0; kh < layer->kernel_size; kh++) {
                        for (int kw = 0; kw < layer->kernel_size; kw++) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            int input_index = ic * (layer->in_height * layer->in_width) + ih * layer->in_width + iw;
                            int weight_index = oc * (layer->in_channels * layer->kernel_size * layer->kernel_size)
                                               + ic * (layer->kernel_size * layer->kernel_size)
                                               + kh * layer->kernel_size + kw;
                            layer->dweights[weight_index] += layer->input[input_index] * grad;
                            dinput[input_index] += layer->weights[weight_index] * grad;
                        }
                    }
                }
            }
        }
    }
}