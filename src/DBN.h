//
// Created by fib1123 on 2015-12-13.
//

#ifndef DEEPBELIEFNETWORK_DBN_H
#define DEEPBELIEFNETWORK_DBN_H
#ifdef __cplusplus
extern "C" {
#endif

	#include "LogisticRegression.h"
	#include "RBM.h"
	#include "HiddenLayer.h"

	typedef struct {
	    int N;
	    int inputDimension;
	    int *hiddenLayerDimensions;
	    int outputDimension;
	    int nrOfLayers;
	    HiddenLayer *sigmoidLayers;
	    RBM *rbmLayers;
	    LogisticRegression logisticRegressionLayer;
	} DBN;

	void loadDBNModel(DBN *, int, int, int *, int, int);
	void buildDBNModel(DBN *, int, int, int *, int, int);
	void freeDBNModel(DBN *);
	void pretrainDBN(DBN *, int *, double, int, int);
	void finetuneDBN(DBN *, int *, int *, double, int);
	void predictDBN(DBN *, int *, double *);
	void persistWeights(DBN*);
#ifdef __cplusplus
}
#endif
#endif //DEEPBELIEFNETWORK_DBN_H
