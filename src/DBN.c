//
// Created by fib1123 on 2015-12-13.
//

#include <stdio.h>
#include <stdlib.h>
#include "DBN.h"
#include "utils.h"
#include "HiddenLayer.h"
#include "RBM.h"					// here
#include "LogisticRegression.h"


static double tabledW[100000000];

void buildDBNModel(DBN *this, int N, \
                    int n_ins, int *hidden_layer_sizes, int n_outs, int n_layers) {
    int i, input_size;

    this->N = N;
    this->inputDimension = n_ins;
    this->hiddenLayerDimensions = hidden_layer_sizes;
    this->outputDimension = n_outs;
    this->nrOfLayers = n_layers;

    this->sigmoidLayers = (HiddenLayer *)malloc(sizeof(HiddenLayer) * n_layers);
    this->rbmLayers = (RBM *)malloc(sizeof(RBM) * n_layers);

    // construct multi-layer
    for(i=0; i<n_layers; i++) {
        if(i == 0) {
            input_size = n_ins;
        } else {
            input_size = hidden_layer_sizes[i-1];
        }

        // construct sigmoid_layer
        buildHiddenLayerModel(&(this->sigmoidLayers[i]), \
                           N, input_size, hidden_layer_sizes[i], NULL, NULL);

        // construct rbm_layer
        buildRBMModel(&(this->rbmLayers[i]), N, input_size, hidden_layer_sizes[i], \
                   this->sigmoidLayers[i].W, this->sigmoidLayers[i].b, NULL);

    }

    // layer for output using LogisticRegression
    buildLogisticRegressionModel(&(this->logisticRegressionLayer), \
                                N, hidden_layer_sizes[n_layers-1], n_outs);

}

void loadLayer(char *fileName, double **W, int n, int m) {
    FILE *f = fopen(fileName, "r");

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fread(&W[i][j], sizeof(double), 1, f);
        }
    }
    fclose(f);
}

void loadB(char *fileName, double *b, int n) {
    FILE *f = fopen(fileName, "r");

    double tabledB[n];
    fread(tabledB, sizeof(double), (size_t) n, f);

    for (int i = 0; i < n; ++i) {
        tabledB[i] = b[i];
    }

    fclose(f);
}

void loadWeights(DBN* dbn) {
    char fileName[16];
    for (int i=0; i<dbn->nrOfLayers; i++) {

        sprintf(fileName, "b%d", i);
        loadB(fileName, dbn->sigmoidLayers[i].b, dbn->sigmoidLayers[i].outputDimension);

        sprintf(fileName, "layer%d", i);
        loadLayer(fileName, dbn->sigmoidLayers[i].W, dbn->sigmoidLayers[i].outputDimension,
                    dbn->sigmoidLayers[i].inputDimension);

    }

    sprintf(fileName, "lrW");
    loadLayer(fileName, dbn->logisticRegressionLayer.W, dbn->logisticRegressionLayer.outputDimension,
                dbn->logisticRegressionLayer.inputDimension);

    sprintf(fileName, "lrB");
    loadB(fileName, dbn->logisticRegressionLayer.b, dbn->logisticRegressionLayer.outputDimension);
}

void loadDBNModel(DBN *this, int N, \
                    int n_ins, int *hidden_layer_sizes, int n_outs, int n_layers) {
    int i, input_size;

    this->N = N;
    this->inputDimension = n_ins;
    this->hiddenLayerDimensions = hidden_layer_sizes;
    this->outputDimension = n_outs;
    this->nrOfLayers = n_layers;

    this->sigmoidLayers = (HiddenLayer *)malloc(sizeof(HiddenLayer) * n_layers);
    this->rbmLayers = (RBM *)malloc(sizeof(RBM) * n_layers);

    // construct multi-layer
    for(i=0; i<n_layers; i++) {
        if(i == 0) {
            input_size = n_ins;
        } else {
            input_size = hidden_layer_sizes[i-1];
        }

        // construct sigmoid_layer
        buildHiddenLayerModel(&(this->sigmoidLayers[i]), \
                           N, input_size, hidden_layer_sizes[i], NULL, NULL);

        // construct rbm_layer
        buildRBMModel(&(this->rbmLayers[i]), N, input_size, hidden_layer_sizes[i], \
                   this->sigmoidLayers[i].W, this->sigmoidLayers[i].b, NULL);

    }

    // layer for output using LogisticRegression
    buildLogisticRegressionModel(&(this->logisticRegressionLayer), \
                                N, hidden_layer_sizes[n_layers-1], n_outs);

    loadWeights(this);
}

void freeDBNModel(DBN *this) {
    int i;
    for(i=0; i<this->nrOfLayers; i++) {
        freeHiddenLayerModel(&(this->sigmoidLayers[i]));
        freeRBMModel(&(this->rbmLayers[i]));
    }
    freeLogisticRegressionModel(&(this->logisticRegressionLayer));
    free(this->sigmoidLayers);
    free(this->rbmLayers);
}

void pretrainDBN(DBN *this, int *input, double lr, int k, int epochs) {

    printf("\n1) Pretraining\n\n");
    int i, j, l, m, n, epoch;

    int *layer_input;
    int prev_layer_input_size;
    int *prev_layer_input;

    int *train_X = (int *)malloc(sizeof(int) * this->inputDimension);

    for(i=0; i<this->nrOfLayers; i++) { // layer-wise

        printf("Layer: %d\n", i);
        for(epoch = 0; epoch < epochs; ++epoch) { // training epochs

//            printf("\nTraining epoch %d", epoch);
            for(n=0; n<this->N; n++) { // input x1...xN

                // initial input
                for(m=0; m<this->inputDimension; m++) train_X[m] = input[n * this->inputDimension + m];

                layer_input = (int *)malloc(sizeof(int) * this->inputDimension);
                for(j=0; j<this->inputDimension; j++) layer_input[j] = train_X[j];

                // layer input
                for(l=1; l<=i; l++) {
                    if(l == 1) prev_layer_input_size = this->inputDimension;
                    else prev_layer_input_size = this->hiddenLayerDimensions[l - 2];

                    prev_layer_input = (int *)malloc(sizeof(int) * prev_layer_input_size);
                    for(j=0; j<prev_layer_input_size; j++) prev_layer_input[j] = layer_input[j];
                    free(layer_input);

                    layer_input = (int *)malloc(sizeof(int) * this->hiddenLayerDimensions[l - 1]);

                    sample_h_given_v_HiddenLayer(&(this->sigmoidLayers[l - 1]), \
                                     prev_layer_input, layer_input);
                    free(prev_layer_input);
                }

                contrastiveDivergence(&(this->rbmLayers[i]), layer_input, lr, k);
            }

        }
    }

    free(train_X);
    free(layer_input);
}

void finetuneDBN(DBN *this, int *input, int *label, double lr, int epochs) {

    printf("\n\n2) Finetuning\n\n");


    int i, j, m, n, epoch;

    int *layer_input;
    // int prev_layer_input_size;
    int *prev_layer_input;

    int *train_X = (int *)malloc(sizeof(int) * this->inputDimension);
    int *train_Y = (int *)malloc(sizeof(int) * this->outputDimension);


    for(epoch=0; epoch<epochs; epoch++) {
        if (epoch%5 == 0) {
            persistWeights(this);
        }
//        printf("Training epoch %d\n", epoch);

        for(n=0; n<this->N; n++) { // input x1...xN

            // initial input
            for(m=0; m<this->inputDimension; m++) train_X[m] = input[n * this->inputDimension + m];
            for(m=0; m<this->outputDimension; m++) train_Y[m] = label[n * this->outputDimension + m];

            // layer input
            for(i=0; i<this->nrOfLayers; i++) {
                if(i == 0) {
                    prev_layer_input = (int *)malloc(sizeof(int) * this->inputDimension);
                    for(j=0; j<this->inputDimension; j++) prev_layer_input[j] = train_X[j];
                } else {
                    prev_layer_input = (int *)malloc(sizeof(int) * this->hiddenLayerDimensions[i - 1]);
                    for(j=0; j<this->hiddenLayerDimensions[i - 1]; j++) prev_layer_input[j] = layer_input[j];
                    free(layer_input);
                }


                layer_input = (int *)malloc(sizeof(int) * this->hiddenLayerDimensions[i]);
                sample_h_given_v_HiddenLayer(&(this->sigmoidLayers[i]), \
                                     prev_layer_input, layer_input);
                free(prev_layer_input);
            }

            train(&(this->logisticRegressionLayer), layer_input, train_Y, lr);
        }
        // lr *= 0.95;
    }

    free(layer_input);
    free(train_X);
    free(train_Y);
}

void predictDBN(DBN *this, int *x, double *y) {
    int i, j, k;
    for (i=0; i<10; i++) {
        y[i] = 0;
    }
    double *layer_input;
    // int prev_layer_input_size;
    double *prev_layer_input;

    double linear_output;
    //TODO print digit
//    int kx, ky;
//    for (ky = 0; ky<28; ky++) {
//        for (kx = 0; kx<28; kx++) {
//            if (x[28*kx + ky] > 0) {
//                printf("O");
//            } else {
//                printf(" ");
//            }
//            // printf("%d", x[kx*ky]/255);
//        }
//        printf("\n");
//    }

    prev_layer_input = (double *)malloc(sizeof(double) * this->inputDimension);
    for(j=0; j<this->inputDimension; j++) prev_layer_input[j] = x[j];

    // layer activation
    for(i=0; i<this->nrOfLayers; i++) {
        layer_input = (double *)malloc(sizeof(double) * this->sigmoidLayers[i].outputDimension);

        for(k=0; k<this->sigmoidLayers[i].outputDimension; k++) {
            linear_output = 0.0;

            for(j=0; j<this->sigmoidLayers[i].inputDimension; j++) {
                linear_output += this->sigmoidLayers[i].W[k][j] * prev_layer_input[j];
            }
            linear_output += this->sigmoidLayers[i].b[k];
            layer_input[k] = sigmoid(linear_output);
        }
        free(prev_layer_input);

        if(i < this->nrOfLayers - 1) {
            prev_layer_input = (double *)malloc(sizeof(double) * this->sigmoidLayers[i].outputDimension);
            for(j=0; j<this->sigmoidLayers[i].outputDimension; j++) prev_layer_input[j] = layer_input[j];
            free(layer_input);
        }

    }

    for(i=0; i<this->logisticRegressionLayer.outputDimension; i++) {
        y[i] = 0;
        for(j=0; j<this->logisticRegressionLayer.inputDimension; j++) {
            y[i] += this->logisticRegressionLayer.W[i][j] * layer_input[j];
        }
        y[i] += this->logisticRegressionLayer.b[i];
    }

    //TODO print digit output
//    for (i=0; i<10; i++) {
//        printf("%.2f ", y[i]);
//    }
//    printf("\n-----------------------------\n");

    softmax(&(this->logisticRegressionLayer), y);

    free(layer_input);
}

void persistLayer(char *fileName, double **W, int n, int m) {
    FILE *f = fopen(fileName, "wb");

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fwrite(&W[i][j], sizeof(double), (size_t) 1, f);
        }
    }

    fclose(f);
}

void persistB(char *fileName, double *b, int n) {
    FILE *f = fopen(fileName, "wb");

    double tabledB[n];
    for (int i = 0; i < n; ++i) {
        tabledB[i] = b[i];
    }

    fwrite(tabledB, sizeof(double), (size_t) n, f);
    fclose(f);
}

void persistWeights(DBN* dbn) {

    char fileName[16];
    for (int i=0; i<dbn->nrOfLayers; i++) {

        sprintf(fileName, "b%d", i);
        persistB(fileName, dbn->sigmoidLayers[i].b, dbn->sigmoidLayers[i].outputDimension);

        sprintf(fileName, "layer%d", i);
        persistLayer(fileName, dbn->sigmoidLayers[i].W, dbn->sigmoidLayers[i].outputDimension,
                       dbn->sigmoidLayers[i].inputDimension);

    }

    sprintf(fileName, "lrW");
    persistLayer(fileName, dbn->logisticRegressionLayer.W, dbn->logisticRegressionLayer.outputDimension,
                   dbn->logisticRegressionLayer.inputDimension);

    sprintf(fileName, "lrB");
    persistB(fileName, dbn->logisticRegressionLayer.b, dbn->logisticRegressionLayer.outputDimension);
}

//void loadWeightsLR() {
////    http://stackoverflow.com/questions/18597685/how-to-write-an-array-to-file-in-c
//}

void test_dbn(void) {
    srand(0);

    int i, j;

    double pretrain_lr = 0.1;
    int pretraining_epochs = 1000;
    int k = 1;
    double finetune_lr = 0.1;
    int finetune_epochs = 500;

    int train_N = 6;
    int test_N = 4;
    int n_ins = 6;
    int n_outs = 2;
    int hidden_layer_sizes[] = {3, 3};
    int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);


    // training data
    int train_X[6][6] = {
            {1, 1, 1, 0, 0, 0},
            {1, 0, 1, 0, 0, 0},
            {1, 1, 1, 0, 0, 0},
            {0, 0, 1, 1, 1, 0},
            {0, 0, 1, 1, 0, 0},
            {0, 0, 1, 1, 1, 0}
    };

    int train_Y[6][2] = {
            {1, 0},
            {1, 0},
            {1, 0},
            {0, 1},
            {0, 1},
            {0, 1}
    };

    // construct DBN
    DBN dbn;
    buildDBNModel(&dbn, train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);

    // pretrain
    pretrainDBN(&dbn, *train_X, pretrain_lr, k, pretraining_epochs);

    // finetune
    finetuneDBN(&dbn, *train_X, *train_Y, finetune_lr, finetune_epochs);

    // test data
    int test_X[4][6] = {
            {1, 1, 0, 0, 0, 0},
            {1, 1, 1, 1, 0, 0},
            {0, 0, 0, 1, 1, 0},
            {0, 0, 1, 1, 1, 0}
    };

    double test_Y[4][2];

    // test
    for(i=0; i<test_N; i++) {
        predictDBN(&dbn, test_X[i], test_Y[i]);
        for(j=0; j<n_outs; j++) {
            printf("%.5f ", test_Y[i][j]);
        }
        printf("\n");
    }

//    persistWeights(&dbn);
    // destruct DBN
    freeDBNModel(&dbn);

}
//
//int main(void) {
//    test_dbn();
//    return 0;
//}

