//
// Created by fib1123 on 2015-12-13.
//

#ifndef DEEPBELIEFNETWORK_HIDDENLAYER_H
#define DEEPBELIEFNETWORK_HIDDENLAYER_H

typedef struct {
    int datasetSize;
    int inputDimension;
    int outputDimension;
    double **W;
    double *b;
} HiddenLayer;

void buildHiddenLayerModel(HiddenLayer*, int, int, int, double**, double*);
void freeHiddenLayerModel(HiddenLayer*);
double hiddenLayerOutput(HiddenLayer*, int*, double*, double);
void sample_h_given_v_HiddenLayer(HiddenLayer *, int *, int *);

#endif //DEEPBELIEFNETWORK_HIDDENLAYER_H
