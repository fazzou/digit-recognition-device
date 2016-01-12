//
// Created by fib1123 on 2015-12-13.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "TrainMnist.h"

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "MnistLoader.h"

#include "DBN.h"

#define TRAINING_SET_SIZE 40000
#define PRETRAINING_EPOCHS 80
#define PRETRAINING_LR 0.1
#define LAYER_SIZE 800

#define FINETUNE_LR 0.1
#define FINETUNE_EPOCHS 1000

// void printImage(double image[28][28]) {

//     for (int i = 0; i < 28; ++i) {
//         for (int j = 0; j < 28; ++j) {
//             printf("%d%.0f ", j, image[i][j]);
//         }
//     }
// }

void printVector(int* vector) {

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            printf("%d\t", vector[i*28+j]);
        }
    }
}

int* vectorize(double image2d[28][28]) {
    int* result = malloc(28* 28 * sizeof(int));
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            result[i*28+j] = (int) (image2d[i][j] * 255);
        }
    }

    return result;
}


 int chooseBest(double resultsOneHotEncoding[10]) {
     for (int j = 0; j < 10; ++j) {
         if (resultsOneHotEncoding[j] > 0.5) {
             return j;
         }
     }
     return -1;
 }

 void trainDBNMnist() {

     mnist_data *data;
     unsigned int cnt;
     int ret;

     if ((ret = mnist_load("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", &data, &cnt))) {
         printf("An error occured: %d\n", ret);
     } else {
         printf("image count: %d\n", cnt);

 //        printImage(data[0].data);
 //        printf("%d", data[0].label);

         int nrOfInputs = TRAINING_SET_SIZE;
         static int inputData[50000][28 * 28];
         for (int k = 0; k < nrOfInputs; ++k) {
             for (int i = 0; i < 28; ++i) {
                 for (int j = 0; j < 28; ++j) {
                     inputData[k][i * 28 + j] = (int) (data[k].data[i][j] * 255);
                 }
             }
         }

         static int y_train[50000][10];
         for (int i = 0; i < nrOfInputs; ++i) {
             for (int j = 0; j < 10; ++j) {
                 if (data[i].label == j) {
                     y_train[i][j] = 1;
                 } else {
                     y_train[i][j] = 0;
                 }
             }
         }

 //        for (int i = 0; i < nrOfInputs; ++i) {
 //            printf("\nlabel: %d\n", data[i].label);
 //            for (int j = 0; j < 10; ++j) {
 //                printf("%f ", y_train[i][j]);
 //            }
 //        }
 //        for (int i = 0; i < 60000; ++i) {
 //            printf("%d ", y_train[i]);
 //        }

         clock_t start = clock();
         // training

         srand(0);


         double pretrain_lr = PRETRAINING_LR;
         int pretraining_epochs = PRETRAINING_EPOCHS;
         int k = 1;
         double finetune_lr = FINETUNE_LR;
         int finetune_epochs = FINETUNE_EPOCHS;

         int train_N = TRAINING_SET_SIZE;
         int n_ins = 28*28;
         int n_outs = 10;
         int hidden_layer_sizes[] = {800, 800, 800};
         int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);

         // construct DBN
         DBN dbn;
//         buildDBNModel(&dbn, train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);

         // pretrain
//         pretrainDBN(&dbn, *inputData, pretrain_lr, k, pretraining_epochs);

//         persistWeights(&dbn);

         // finetune
         loadDBNModel(&dbn, train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);
         finetuneDBN(&dbn, *inputData, *y_train, finetune_lr, finetune_epochs);

 //        persistWeightsLR(&dbn);
         // destruct DBN
         freeDBNModel(&dbn);
         free(data);

         clock_t end = clock();
         float seconds = (float) (end - start) / CLOCKS_PER_SEC;
         printf("Trained for: %f seconds\n", seconds);
     }
 }

 void testDBNMnist() {


     mnist_data *data;
     unsigned int cnt;
     int ret;

     if ((ret = mnist_load("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", &data, &cnt))) {
         printf("An error occured: %d\n", ret);
     } else {
         printf("image count: %d\n", cnt);

 //        printImage(data[0].data);
 //        printf("%d", data[0].label);

         int train_N = TRAINING_SET_SIZE;
         int n_ins = 28*28;
         int n_outs = 10;
         int hidden_layer_sizes[] = {800, 800, 800};
         int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);


         // construct LogisticRegression
         DBN classifier;
         loadDBNModel(&classifier, train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);

         // load testing data
         int testingSetSize = 60000;
         static int inputData[60000][28 * 28];
         for (int k = 0; k < testingSetSize; ++k) {
             for (int i = 0; i < 28; ++i) {
                 for (int j = 0; j < 28; ++j) {
                     inputData[k][i * 28 + j] = (int) (data[k].data[i][j] * 255);
                 }
             }
         }

         // ONE HOT ENCODING
         static int y_train[60000][10];
         for (int i = 0; i < testingSetSize; ++i) {
             for (int j = 0; j < 10; ++j) {
                 if (data[i].label == j) {
                     y_train[i][j] = 1;
                 } else {
                     y_train[i][j] = 0;
                 }
             }
         }

 //        for (int i = 0; i < testingSetSize; ++i) {
 //            printf("\nlabel: %d\n", data[i].label);
 //            for (int j = 0; j < 10; ++j) {
 //                printf("%f ", y_train[i][j]);
 //            }
 //        }
 //        for (int i = 0; i < 60000; ++i) {
 //            printf("%d ", y_train[i]);
 //        }


         double test_Y[60000][10];

         // test
         int valid = 0;
         int result = -1;
 //    for(int i = 0; i < testingSetSize; ++i) {
         for (int i = 50000; i < 60000; ++i) {
             predictDBN(&classifier, inputData[i], test_Y[i]);
 //            for (int j = 0; j < nrOfClasses; ++j) {
 //                printf("%f %d\n", test_Y[i][j], y_train[i][j]);
 //            }
             result = chooseBest(test_Y[i]);
             if (result == data[i].label) {
                 valid++;
             }
 //            printf("%d:\t%d\t%d", i, data[i].label, result);
 //            printf("\n");
         }

         printf("Accuracy: %f", ((float) valid)/10000);

         freeDBNModel(&classifier);
         free(data);
     }
 }


int main(int argc, char **argv)
{
    trainDBNMnist();

//    testDBNMnist();

    return 0;
}
