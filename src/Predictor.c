////
//// Created by fib1123 on 2015-12-15.
////
//
//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//#include "Predictor.h"
//
//#define USE_MNIST_LOADER
//#define MNIST_DOUBLE
//#include "MnistLoader.h"
//#include "DBN.h"
//
//int train_N = 100;
//int n_ins = 28*28;
//int n_outs = 10;
//int hidden_layer_sizes[] = {500, 500, 2000};
//int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);
//
//
//// int main(int argc, char **argv)
//// {
////     mnist_data *data;
////     unsigned int cnt;
////     int ret;
//
////     if ((ret = mnist_load("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", &data, &cnt))) {
////         printf("An error occured: %d\n", ret);
////     } else {
////         printf("image count: %d\n", cnt);
//
//// //        printImage(data[0].data);
//// //        printf("%d", data[0].label);
//
////         int nrOfInputs = 20000;
////         static int inputData[20000][28*28];
////         for (int k = 0; k < nrOfInputs; ++k) {
////             for (int i = 0; i < 28; ++i) {
////                 for (int j = 0; j < 28; ++j) {
////                     inputData[k][i*28+j] = (int) (data[k].data[i][j] * 255);
////                 }
////             }
////         }
//// //        int** inputData;
//// //        inputData = (int**) malloc(60000 * sizeof(int*));
//// //        inputData[0] = (int*) malloc(60000* 28 * 28 * sizeof(int));
//
//// //        for (int i = 0; i < cnt; ++i) {
//// //
//// //            inputData[i] = vectorize(data[i].data);
//// //        }
//
////         static int y_train[20000][10];
////         for (int i = 0; i < nrOfInputs; ++i) {
////             for (int j = 0; j < 10; ++j) {
////                 if (data[i].label == j) {
////                     y_train[i][j] = 1;
////                 } else {
////                     y_train[i][j] = 0;
////                 }
////             }
////         }
//
//// //        for (int i = 0; i < nrOfInputs; ++i) {
//// //            printf("\nlabel: %d\n", data[i].label);
//// //            for (int j = 0; j < 10; ++j) {
//// //                printf("%f ", y_train[i][j]);
//// //            }
//// //        }
//// //        for (int i = 0; i < 60000; ++i) {
//// //            printf("%d ", y_train[i]);
//// //        }
//
////         clock_t start = clock();
////         // training
//
////         srand(0);
//
////         int i, j;
//
////         double pretrain_lr = 0.1;
////         int pretraining_epochs = 10;
////         int k = 1;
////         double finetune_lr = 0.1;
////         int finetune_epochs = 20;
//
////         int train_N = nrOfInputs;
////         int n_ins = 28*28;
////         int n_outs = 10;
////         int hidden_layer_sizes[] = {500, 500, 2000};
////         int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);
//
//
////         // construct DBN
////         DBN dbn;
////         loadDBNModel(&dbn, train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);
////         printf("loaded");
//
//// //        // pretrain
//// //        pretrainDBN(&dbn, *inputData, pretrain_lr, k, pretraining_epochs);
//// //
//// //        persistWeights(&dbn);
//
////         // finetune
////         finetuneDBN(&dbn, *inputData, *y_train, finetune_lr, finetune_epochs);
//
////         persistWeights(&dbn);
////         // destruct DBN
////         freeDBNModel(&dbn);
//
////         free(data);
//
////         clock_t end = clock();
////         float seconds = (float)(end - start) / CLOCKS_PER_SEC;
////         printf("Trained for: %f seconds", seconds);
////     }
//
////     return 0;
//// }
