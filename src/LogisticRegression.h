#ifndef DEEPBELIEFNETWORK_LOGISTICREGRESSION_H
#define DEEPBELIEFNETWORK_LOGISTICREGRESSION_H


#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */

typedef struct {
    int datasetSize;
    int inputDimension;
    int outputDimension;
    double **W;
    double *b;
} LogisticRegression;

void buildLogisticRegressionModel(LogisticRegression*, int, int, int);
void freeLogisticRegressionModel(LogisticRegression*);
void train(LogisticRegression*, int*, int*, double);
void softmax(LogisticRegression*, double*);
void predict(LogisticRegression*, int*, double*);
void testLogisticRegression(void);
void loadWeightsLR(LogisticRegression*);
int chooseBest(double[10]);
void printImageInt(int[28][28]);

#  ifdef __cplusplus
}
#  endif /* __cplusplus */

#endif //DEEPBELIEFNETWORK_LOGISTICREGRESSION_H
