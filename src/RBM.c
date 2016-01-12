//
// Created by fib1123 on 2015-12-13.
//

#include <stdlib.h>
#include <stdio.h>
#include "RBM.h"
#include "utils.h"

void buildRBMModel(RBM *this, int N, int n_visible, int n_hidden, \
                    double **W, double *hbias, double *vbias) {
    int i, j;
    double a = 1.0 / n_visible;

    this->datasetSize = N;
    this->visibleDimension = n_visible;
    this->hiddenDimension = n_hidden;

    if(W == NULL) {
        this->W = (double **)malloc(sizeof(double*) * n_hidden);
        this->W[0] = (double *)malloc(sizeof(double) * n_visible * n_hidden);
        for(i=0; i<n_hidden; i++) this->W[i] = this->W[0] + i * n_visible;

        for(i=0; i<n_hidden; i++) {
            for(j=0; j<n_visible; j++) {
                this->W[i][j] = uniform(-a, a);
            }
        }
    } else {
        this->W = W;
    }

    if(hbias == NULL) {
        this->hBias = (double *)malloc(sizeof(double) * n_hidden);
        for(i=0; i<n_hidden; i++) this->hBias[i] = 0;
    } else {
        this->hBias = hbias;
    }

    if(vbias == NULL) {
        this->vBias = (double *)malloc(sizeof(double) * n_visible);
        for(i=0; i<n_visible; i++) this->vBias[i] = 0;
    } else {
        this->vBias = vbias;
    }
}

void freeRBMModel(RBM *this) {
    // free(this->W[0]);
    // free(this->W);
    // free(this->hBias);
    free(this->vBias);
}

void contrastiveDivergence(RBM *this, int *input, double lr, int k) {
    int i, j, step;

    double *ph_mean = (double *) malloc(sizeof(double) * this->hiddenDimension);
    int *ph_sample = (int *) malloc(sizeof(int) * this->hiddenDimension);
    double *nv_means = (double *) malloc(sizeof(double) * this->visibleDimension);
    int *nv_samples = (int *) malloc(sizeof(int) * this->visibleDimension);
    double *nh_means = (double *) malloc(sizeof(double) * this->hiddenDimension);
    int *nh_samples = (int *) malloc(sizeof(int) * this->hiddenDimension);

    /* CD-k */
    sample_h_given_v_RBM(this, input, ph_mean, ph_sample);

    for(step=0; step<k; step++) {
        if(step == 0) {
            gibbs_hvh_RBM(this, ph_sample, nv_means, nv_samples, nh_means, nh_samples);
        } else {
            gibbs_hvh_RBM(this, nh_samples, nv_means, nv_samples, nh_means, nh_samples);
        }
    }

    for(i=0; i<this->hiddenDimension; i++) {
        for(j=0; j<this->visibleDimension; j++) {
            // this->W[i][j] += lr * (ph_sample[i] * input[j] - nh_means[i] * nv_samples[j]) / this->N;
            this->W[i][j] += lr * (ph_mean[i] * input[j] - nh_means[i] * nv_samples[j]) / this->datasetSize;
        }
        this->hBias[i] += lr * (ph_sample[i] - nh_means[i]) / this->datasetSize;
    }

    for(i=0; i<this->visibleDimension; i++) {
        this->vBias[i] += lr * (input[i] - nv_samples[i]) / this->datasetSize;
    }


    free(ph_mean);
    free(ph_sample);
    free(nv_means);
    free(nv_samples);
    free(nh_means);
    free(nh_samples);
}


void sample_h_given_v_RBM(RBM *this, int *v0_sample, double *mean, int *sample) {
    int i;
    for(i=0; i<this->hiddenDimension; i++) {
        mean[i] = propup_RBM(this, v0_sample, this->W[i], this->hBias[i]);
        sample[i] = binomial(1, mean[i]);
    }
}

void sample_v_given_h_RBM(RBM *this, int *h0_sample, double *mean, int *sample) {
    int i;
    for(i=0; i<this->visibleDimension; i++) {
        mean[i] = propdown_RBM(this, h0_sample, i, this->vBias[i]);
        sample[i] = binomial(1, mean[i]);
    }
}

double propup_RBM(RBM *this, int *v, double *w, double b) {
    int j;
    double pre_sigmoid_activation = 0.0;
    for(j=0; j<this->visibleDimension; j++) {
        pre_sigmoid_activation += w[j] * v[j];
    }
    pre_sigmoid_activation += b;
    return sigmoid(pre_sigmoid_activation);
}

double propdown_RBM(RBM *this, int *h, int i, double b) {
    int j;
    double pre_sigmoid_activation = 0.0;

    for(j=0; j<this->hiddenDimension; j++) {
        pre_sigmoid_activation += this->W[j][i] * h[j];
    }
    pre_sigmoid_activation += b;
    return sigmoid(pre_sigmoid_activation);
}

void gibbs_hvh_RBM(RBM *this, int *h0_sample, double *nv_means, int *nv_samples, \
                   double *nh_means, int *nh_samples) {
    sample_v_given_h_RBM(this, h0_sample, nv_means, nv_samples);
    sample_h_given_v_RBM(this, nv_samples, nh_means, nh_samples);
}

void reconstructRBM(RBM *this, int *v, double *reconstructed_v) {
    int i, j;
    double *h = (double *)malloc(sizeof(double) * this->hiddenDimension);
    double pre_sigmoid_activation;

    for(i=0; i<this->hiddenDimension; i++) {
        h[i] = propup_RBM(this, v, this->W[i], this->hBias[i]);
    }

    for(i=0; i<this->visibleDimension; i++) {
        pre_sigmoid_activation = 0.0;
        for(j=0; j<this->hiddenDimension; j++) {
            pre_sigmoid_activation += this->W[j][i] * h[j];
        }
        pre_sigmoid_activation += this->vBias[i];

        reconstructed_v[i] = sigmoid(pre_sigmoid_activation);
    }

    free(h);
}

void test_rbm(void) {
    srand(0);

    int i, j, epoch;

    double learning_rate = 0.1;
    int training_epochs = 1000;
    int k = 1;

    int train_N = 6;
    int test_N = 2;
    int n_visible = 6;
    int n_hidden = 3;

    // training data
    int train_X[6][6] = {
            {1, 1, 1, 0, 0, 0},
            {1, 0, 1, 0, 0, 0},
            {1, 1, 1, 0, 0, 0},
            {0, 0, 1, 1, 1, 0},
            {0, 0, 1, 0, 1, 0},
            {0, 0, 1, 1, 1, 0}
    };

    // construct RBM
    RBM rbm;
    buildRBMModel(&rbm, train_N, n_visible, n_hidden, NULL, NULL, NULL);

    // train
    for(epoch=0; epoch<training_epochs; epoch++) {
        for(i=0; i<train_N; i++) {
            contrastiveDivergence(&rbm, train_X[i], learning_rate, k);
        }
    }


    // test data
    int test_X[2][6] = {
            {1, 1, 0, 0, 0, 0},
            {0, 0, 0, 1, 1, 0}
    };
    double reconstructed_X[2][6];

    // test
    for(i=0; i<test_N; i++) {
        reconstructRBM(&rbm, test_X[i], reconstructed_X[i]);
        for(j=0; j<n_visible; j++) {
            printf("%.5f ", reconstructed_X[i][j]);
        }
        printf("\n");
    }


    // destruct RBM
    freeRBMModel(&rbm);
}



//int main(void) {
//    test_rbm();
//
//    return 0;
//}