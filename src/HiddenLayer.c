//
// Created by fib1123 on 2015-12-13.
//

#include <stdlib.h>
#include "HiddenLayer.h"
#include "utils.h"

void buildHiddenLayerModel(HiddenLayer* hiddenLayer, int datasetSize, int inputDimension, int outputDimension, \
                            double** W, double* b) {

    double a = 1.0 / inputDimension;

    hiddenLayer->datasetSize = datasetSize;
    hiddenLayer->inputDimension = inputDimension;
    hiddenLayer->outputDimension = outputDimension;

    if(W == NULL) {
        hiddenLayer->W = (double**) malloc(outputDimension * sizeof(double*));
        hiddenLayer->W[0] = (double*) malloc(inputDimension * outputDimension * sizeof(double));
        for (int i = 0; i < outputDimension; i++) {
            hiddenLayer->W[i] = hiddenLayer->W[0] + i * inputDimension;
        }

        for (int i=0; i < outputDimension; ++i) {
            for (int j=0; j < inputDimension; ++j) {
                hiddenLayer->W[i][j] = uniform(-a, a);
            }
        }
    } else {
        hiddenLayer->W = W;
    }

    if (b == NULL) {
        hiddenLayer->b = (double*) malloc(outputDimension * sizeof(double));
    } else {
        hiddenLayer->b = b;
    }
}

void freeHiddenLayerModel(HiddenLayer *hiddenLayer) {
    free(hiddenLayer->W[0]);
    free(hiddenLayer->W);
    free(hiddenLayer->b);
}

double hiddenLayerOutput(HiddenLayer *hiddenLayer, int *input, double *w, double b) {

    double linearOutput = 0.0;
    for (int j = 0; j < hiddenLayer->inputDimension; ++j) {
        linearOutput += w[j] * input[j];
    }
    linearOutput += b;

    return sigmoid(linearOutput);
}

void sample_h_given_v_HiddenLayer(HiddenLayer *hiddenLayer, int *input, int *sample) {

    for(int i = 0; i < hiddenLayer->outputDimension; ++i) {
        sample[i] = binomial(1, hiddenLayerOutput(hiddenLayer, input, hiddenLayer->W[i], hiddenLayer->b[i]));
    }
}