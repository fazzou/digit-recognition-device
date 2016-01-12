//
// Created by fib1123 on 2015-12-13.
//

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "TrainMnist.h"

//#define USE_MNIST_LOADER
//#define MNIST_DOUBLE
//#include "MnistLoader.h"

#include "LogisticRegression.h"

#define NR_OF_LABELS 10

void printImage(double image[28][28]) {

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            printf("%.0f ", image[i][j]);
        }
        printf("\n");
    }
}

void printImageInt(int image[28][28]) {

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (image[i][j] == 0) printf(" ");
            else printf("o");
//            printf("%d ", image[i][j]);
        }
        printf("\n");
    }
}

void normalize(int image[28][28]) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (image[i][j] > 0) { image[i][j] = 1; }
            else { image[i][j] = 0; }
        }
    }
}

void buildLogisticRegressionModel(LogisticRegression* logisticRegression, int N, int n_in, int n_out) {
    logisticRegression->datasetSize = N;
    logisticRegression->inputDimension = n_in;
    logisticRegression->outputDimension = n_out;

    logisticRegression->W = (double**) malloc(n_out * sizeof(double*));
    logisticRegression->W[0] = (double*) malloc(n_in * n_out * sizeof(double));

    for (int i = 0; i < n_out; ++i) {
        logisticRegression->W[i] = logisticRegression->W[0] + i * n_in;
    }
    logisticRegression->b = (double*) malloc( n_out * sizeof(double));

    for (int i = 0; i < n_out; ++i) {
        for(int j = 0; j < n_in; ++j) {
            logisticRegression->W[i][j] = 0;
        }
        logisticRegression->b[i] = 0;
    }
}

void freeLogisticRegressionModel(LogisticRegression* logisticRegression) {
    free(logisticRegression->W[0]);
    free(logisticRegression->W);
    free(logisticRegression->b);
}

double dotProduct(int size, double* a, int* b) {
    double result = 0;
    for (int i = 0; i < size; ++i) {
        result = a[i] * b[i];
    }
    return result;
}

//TODO extract to linear algebra util
//double* matrixVectorProduct(int a, )
//double* addVectors()

void train(LogisticRegression* logisticRegression, int* x, int* y, double learningRate) {

    int nrOfTrainingExamples = logisticRegression->datasetSize;
    int inputSize = logisticRegression->inputDimension;
    int outputSize = logisticRegression->outputDimension;
    double** W = logisticRegression->W;
    double* b = logisticRegression->b;

    double* p_y_given_x = (double *) malloc(outputSize * sizeof(double));
    double* dy = (double *) malloc(outputSize * sizeof(double));

    //TODO replace with matrix vector product
    for (int i = 0; i < outputSize; ++i) {
        //TODO change to dot product
//        p_y_given_x[i] = dotProduct(inputDimension, W[i], x) + b[i];
        p_y_given_x[i] = 0;

        for(int j = 0; j < inputSize; ++j) {
            p_y_given_x[i] += W[i][j] * x[j];
        }
        p_y_given_x[i] += b[i];
    }
    softmax(logisticRegression, p_y_given_x);

    for(int i = 0; i < outputSize; ++i) {
        //TODO change to logistic regression loss function
        dy[i] = y[i] - p_y_given_x[i];

        for(int j = 0; j < inputSize; ++j) {
            W[i][j] += learningRate * dy[i] * x[j] / nrOfTrainingExamples;
        }

        logisticRegression->b[i] += learningRate * dy[i] / nrOfTrainingExamples;
    }

    free(p_y_given_x);
    free(dy);
}

void softmax(LogisticRegression* logisticRegression, double* a) {

    int outputSize = logisticRegression->outputDimension;

    double max = 0.0;
    double sum = 0.0;

    for(int i = 0; i < outputSize; i++) if(max < a[i]) max = a[i];

    for(int i = 0; i < outputSize; i++) {
        a[i] = exp(a[i] - max);
        sum += a[i];
    }

    for(int i=0; i < outputSize; i++) a[i] /= sum;
}

void predict(LogisticRegression*logisticRegression, int* x, double* y) {


    int inputSize = logisticRegression->inputDimension;
    int outputSize = logisticRegression->outputDimension;
    double** W = logisticRegression->W;
    double* b = logisticRegression->b;

    for(int i = 0; i < outputSize; ++i) {
        //TODO change to dotProduct
        y[i] = 0;
        for(int j=0; j < inputSize; ++j) {
            y[i] += W[i][j] * x[j];
        }
        y[i] += b[i];

//        y[i] = dotProduct(inputDimension, W[i], x) + b[i];
    }

    softmax(logisticRegression, y);
}


static double tabledW[100000000];
static double tabledW[100000000];

int chooseBest(double resultsOneHotEncoding[10]);

//TODO move to utils
void persistLayerLR(char *fileName, double **W, int n, int m) {
    FILE *f = fopen(fileName, "wb");

//    printf("\n%s\n", fileName);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
//            tabledW[i*n+m] = W[i][j];
//            printf("%f ", W[i][j]);
            fwrite(&W[i][j], sizeof(double), (size_t) 1, f);
        }
//        printf("\n");
    }

    fclose(f);
}

void persistBLR(char *fileName, double *b, int n) {
    FILE *f = fopen(fileName, "wb");

//    printf("\n%s\n", fileName);
    double tabledB[n];
    for (int i = 0; i < n; ++i) {
        tabledB[i] = b[i];
//        printf("%f ",tabledB[i]);
    }
//    printf("\n");

    fwrite(tabledB, sizeof(double), (size_t) n, f);
    fclose(f);
}


void persistWeightsLR(LogisticRegression *logisticRegression) {

    char fileName[16];

    sprintf(fileName, "lrW");
    persistLayerLR(fileName, logisticRegression->W, logisticRegression->outputDimension,
                   logisticRegression->inputDimension);

    sprintf(fileName, "lrB");
    persistBLR(fileName, logisticRegression->b, logisticRegression->outputDimension);
}

void loadLayerLR(char *fileName, double **W, int n, int m) {
    FILE *f = fopen(fileName, "r");

//    fread(tabledW, sizeof(double), (size_t) (n * m), f);

//    printf("\n%s\n", fileName);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fread(&W[i][j], sizeof(double), 1, f);
//            W[i][j] = tabledW[i*n+j];
//            printf("%f ", W[i][j]);
        }
//        printf("\n");
    }
    fclose(f);
}

void loadBLR(char *fileName, double *b, int n) {
    FILE *f = fopen(fileName, "r");

//    double tabledB[n];
    fread(b, sizeof(double), (size_t) n, f);

//    printf("\n%s\n", fileName);
//    for (int i = 0; i < n; ++i) {
//        printf("%f ",tabledB[i]);
//        tabledB[i] = b[i];
//    }
//    printf("\n");

    fclose(f);
}

void loadWeightsLR(LogisticRegression *logisticRegression) {
    char fileName[16];

    sprintf(fileName, "lrW");
    loadLayerLR(fileName, logisticRegression->W, logisticRegression->outputDimension,
                logisticRegression->inputDimension);

    sprintf(fileName, "lrB");
    loadBLR(fileName, logisticRegression->b, logisticRegression->outputDimension);
}

////TODO Remove
//void trainLogisticRegression() {
//
//    double learningRate = 0.1;
//    int nrOfEpochs = 500;
//
//    int trainingSetSize = 6;
//    int inputDimension = 6;
//    int nrOfClasses = 2;
//
//    // training data
//    //INFO keep sizes in sync with variables: trainingSetSize & inputDimension
//    int train_X[6][6] = {
//            {1, 1, 1, 0, 0, 0},
//            {1, 0, 1, 0, 0, 0},
//            {1, 1, 1, 0, 0, 0},
//            {0, 0, 1, 1, 1, 0},
//            {0, 0, 1, 1, 0, 0},
//            {0, 0, 1, 1, 1, 0}
//    };
//
//    //INFO keep sizes in sync with variables: trainingSetSize & inputDimension
//    int train_Y[6][6] = {
//            {1, 0},
//            {1, 0},
//            {1, 0},
//            {0, 1},
//            {0, 1},
//            {0, 1}
//    };
//
//    // construct LogisticRegression
//    LogisticRegression classifier;
//    buildLogisticRegressionModel(&classifier, trainingSetSize, inputDimension, nrOfClasses);
//
//    // train
//    for(int epoch=0; epoch < nrOfEpochs; epoch++) {
//        for(int i=0; i < trainingSetSize; i++) {
//            train(&classifier, train_X[i], train_Y[i], learningRate);
//        }
////        learningRate *= 0.99;
//    }
//
//    persistWeightsLR(&classifier);
//
//    freeLogisticRegressionModel(&classifier);
//}
//
////TODO Remove
//void trainLogisticRegressionMnist() {
//
//    mnist_data *data;
//    unsigned int cnt;
//    int ret;
//
//    if ((ret = mnist_load("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", &data, &cnt))) {
//        printf("An error occured: %d\n", ret);
//    } else {
//        printf("image count: %d\n", cnt);
//
////        printImage(data[0].data);
////        printf("%d", data[0].label);
//
//        int nrOfInputs = 50000;
//        static int inputData[50000][28 * 28];
//        for (int k = 0; k < nrOfInputs; ++k) {
//            for (int i = 0; i < 28; ++i) {
//                for (int j = 0; j < 28; ++j) {
//                    inputData[k][i * 28 + j] = (int) (data[k].data[i][j] * 255);
//                }
//            }
//        }
//
//        static int y_train[50000][10];
//        for (int i = 0; i < nrOfInputs; ++i) {
//            for (int j = 0; j < 10; ++j) {
//                if (data[i].label == j) {
//                    y_train[i][j] = 1;
//                } else {
//                    y_train[i][j] = 0;
//                }
//            }
//        }
//
////        for (int i = 0; i < nrOfInputs; ++i) {
////            printf("\nlabel: %d\n", data[i].label);
////            for (int j = 0; j < 10; ++j) {
////                printf("%f ", y_train[i][j]);
////            }
////        }
////        for (int i = 0; i < 60000; ++i) {
////            printf("%d ", y_train[i]);
////        }
//
//        clock_t start = clock();
//        // training
//
//        srand(0);
//
//        double learningRate = 0.1;
//        int nrOfEpochs = 100;
//
//        int trainingSetSize = nrOfInputs;
//        int inputDimension = 28 * 28;
//        int nrOfClasses = 10;
//
//        // construct LogisticRegression
//        LogisticRegression classifier;
//        buildLogisticRegressionModel(&classifier, trainingSetSize, inputDimension, nrOfClasses);
//
//        // train
//        for (int epoch = 0; epoch < nrOfEpochs; epoch++) {
//            printf("%d", epoch);
//            for (int i = 0; i < trainingSetSize; i++) {
//                train(&classifier, inputData[i], y_train[i], learningRate);
//            }
////        learningRate *= 0.99;
//        }
//
//        persistWeightsLR(&classifier);
//
//        freeLogisticRegressionModel(&classifier);
//        free(data);
//
//        clock_t end = clock();
//        float seconds = (float) (end - start) / CLOCKS_PER_SEC;
//        printf("Trained for: %f seconds", seconds);
//    }
//}
//
////TODO Remove
//void testLogisticRegression() {
//
//    int trainingSetSize = 6;
//    int testingSetSize = 3;
//    int inputDimension = 6;
//    int nrOfClasses = 2;
//
//    // construct LogisticRegression
//    LogisticRegression classifier;
//    buildLogisticRegressionModel(&classifier, trainingSetSize, inputDimension, nrOfClasses);
//    loadWeightsLR(&classifier);
//
//    // test data
//    //INFO keep sizes in sync with variables: testingSetSize & inputDimension
//    int test_X[3][6] = {
//            {1, 0, 1, 0, 0, 0},
//            {0, 0, 1, 1, 1, 0},
//            {1, 1, 1, 1, 0, 0}
//    };
//
//    //INFO keep sizes in sync with variables: testingSetSize
//    double test_Y[3][2];
//
//    // test
//    for(int i = 0; i < testingSetSize; ++i) {
//        predict(&classifier, test_X[i], test_Y[i]);
//        for (int j = 0; j < nrOfClasses; ++j) {
//            printf("%f ", test_Y[i][j]);
//        }
//        printf("\n");
//    }
//
//    freeLogisticRegressionModel(&classifier);
//}
//
//int chooseBest(double resultsOneHotEncoding[10]) {
//    for (int j = 0; j < 10; ++j) {
//        if (resultsOneHotEncoding[j] > 0.5) {
//            return j;
//        }
//    }
//    return -1;
//}
//
//
////TODO Remove
//void testLogisticRegressionMnist() {
//
//
//    mnist_data *data;
//    unsigned int cnt;
//    int ret;
//
//    if ((ret = mnist_load("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", &data, &cnt))) {
//        printf("An error occured: %d\n", ret);
//    } else {
//        printf("image count: %d\n", cnt);
//
////        printImage(data[0].data);
////        printf("%d", data[0].label);
//
//        int trainingSetSize = 50000;
//        int inputDimension = 28 * 28;
//        int nrOfClasses = 10;
//
//        // construct LogisticRegression
//        LogisticRegression classifier;
//        buildLogisticRegressionModel(&classifier, trainingSetSize, inputDimension, nrOfClasses);
//        loadWeightsLR(&classifier);
//
//
//        int testingSetSize = 60000;
//        static int inputData[60000][28 * 28];
//        for (int k = 0; k < testingSetSize; ++k) {
//            for (int i = 0; i < 28; ++i) {
//                for (int j = 0; j < 28; ++j) {
//                    inputData[k][i * 28 + j] = (int) (data[k].data[i][j] * 255);
//                }
//            }
//        }
//
//        static int y_train[60000][10];
//        for (int i = 0; i < testingSetSize; ++i) {
//            for (int j = 0; j < 10; ++j) {
//                if (data[i].label == j) {
//                    y_train[i][j] = 1;
//                } else {
//                    y_train[i][j] = 0;
//                }
//            }
//        }
//
////        for (int i = 0; i < testingSetSize; ++i) {
////            printf("\nlabel: %d\n", data[i].label);
////            for (int j = 0; j < 10; ++j) {
////                printf("%f ", y_train[i][j]);
////            }
////        }
////        for (int i = 0; i < 60000; ++i) {
////            printf("%d ", y_train[i]);
////        }
//
//
//        double test_Y[60000][10];
//
//        // test
//        int valid = 0;
//        int misclassified[NR_OF_LABELS];
//        int result = -1;
//
//        for (int i = 0; i < NR_OF_LABELS; ++i) {
//            misclassified[i] = 0;
//        }
//
////    for(int i = 0; i < testingSetSize; ++i) {
//        for (int i = 50000; i < 60000; ++i) {
//            predict(&classifier, inputData[i], test_Y[i]);
////            for (int j = 0; j < nrOfClasses; ++j) {
////                printf("%f %d\n", test_Y[i][j], y_train[i][j]);
////            }
//            result = chooseBest(test_Y[i]);
//            if (result == data[i].label) {
//                valid++;
//            } else {
////                if (data[i].label == 3) {
////                    printImageInt(inputData[i]);
////                }
//
////            printf("%d:\t%d\t%d", i, data[i].label, result);
////            printf("\n");
//            ++misclassified[data[i].label];
//        }
//    }
//
//        for (int i = 0; i < NR_OF_LABELS; ++i) {
//            printf("Misclassified with %d:\t%d\n", i, misclassified[i]);
//        }
//
//        printf("Accuracy: %f", ((float) valid)/10000);
//        freeLogisticRegressionModel(&classifier);
//
//        free(data);
//    }
//}

//int main() {
//    trainLogisticRegressionMnist();
//
//    testLogisticRegressionMnist();
//
//    return 0;
//}