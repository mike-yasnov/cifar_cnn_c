#ifndef SOFTMAX_H
#define SOFTMAX_H

float softmax(const float *input, int length, int label, float *probabilities);
void softmax_backward(const float *probabilities, int label, int length, float *dout);

#endif 