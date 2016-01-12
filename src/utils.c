//
// Created by fib1123 on 2015-12-13.
//

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"

double uniform(double min, double max) {
    return (max - min) * rand() / (RAND_MAX + 1.0) + min;
}

int binomial(int n, double p) {
    if (p < 0 || p > 1) return 0;

    int c = 0;
    double r;

    for(int i = 0; i < n; ++i) {
        r = rand() / (RAND_MAX + 1.0);
        if (r < p) c++;
    }

    return c;
}

double sigmoid(double x) {
//    printf("%.1f", x);
    return 1.0 / (1.0 + exp(-x));
}
