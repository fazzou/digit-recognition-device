//
// Created by fib1123 on 2015-12-13.
//

#ifndef DEEPBELIEFNETWORK_RBM_H
#define DEEPBELIEFNETWORK_RBM_H

typedef struct {
    int datasetSize;
    int visibleDimension;
    int hiddenDimension;
    double **W;
    double *hBias;
    double *vBias;
} RBM;

void buildRBMModel(RBM*, int, int, int, double**, double*, double*);
void freeRBMModel(RBM *);
void contrastiveDivergence(RBM *, int *, double, int);
void sample_h_given_v_RBM(RBM *, int *, double *, int *);
void sample_v_given_h_RBM(RBM *, int *, double *, int *);
double propup_RBM(RBM *, int *, double *, double);
double propdown_RBM(RBM *, int *, int, double);
void gibbs_hvh_RBM(RBM *, int *, double *, int *, double *, int *);
void reconstructRBM(RBM *, int *, double *);

#endif //DEEPBELIEFNETWORK_RBM_H
