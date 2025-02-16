#ifndef SOFTMAX_H
#define SOFTMAX_H

float softmax(float *input, int length, int label, float *probabilities);
void softmax_backward(float *probabilities, int label, int length, float *dout);

#endif 