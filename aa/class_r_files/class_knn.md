# kNN

``` r
# Load libraries
library(e1071)
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

## Import Data

``` r
# Read training and testing data
train <- read.csv("./../data/classification_data/intermediates/train.csv")
test <- read.csv("./../data/classification_data/intermediates/test.csv")
```

## Model Training

``` r
ks <- c(3, 5, 9, 15, 23)
initial_accuracy <- 0
for (k in ks){
  print(k)
  knn.model <- knn3(as.factor(Rating)~., data=train, k=k)
  result <- predict(knn.model, test, type="class")
  sum(is.na(result))
  sum(is.na(test$Rating))
  accuracy <- mean(as.factor(result)==as.factor(test$Rating))
  print(accuracy)
  if (initial_accuracy < accuracy){
    initial_accuracy <- accuracy
    chosen_k <- k
  }
  print("\n")
}
```

    ## [1] 3
    ## [1] 0.3678087
    ## [1] "\n"
    ## [1] 5
    ## [1] 0.3096315
    ## [1] "\n"
    ## [1] 9
    ## [1] 0.2281836
    ## [1] "\n"
    ## [1] 15
    ## [1] 0.1932773
    ## [1] "\n"
    ## [1] 23
    ## [1] 0.1797027
    ## [1] "\n"

``` r
print(chosen_k)
```

    ## [1] 3

``` r
# Retrain the model with chosen value of k
knn.model <- knn3(as.factor(Rating)~., data=train, k=chosen_k)
knn.model
```

    ## 3-nearest neighbor model
    ## Training set outcome distribution:
    ## 
    ##   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
    ##  72  53 154 197 378 649 577 660 737 559 433 368 348 330 245 230 121  54   9  15 
    ##  21 
    ##   9

## Model Validation

``` r
# Use the new model for prediction
result <- predict(knn.model, test, type="class")

# Get the confusion matrix with the predicted results
confusionMatrix(as.factor(result), as.factor(test$Rating))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
    ##         1   6  1  1  0  0  0  5  1  1  2  0  0  1  1  0  0  0  0  0  0  0
    ##         2   3 13  1  1  1  2  0  1  1  0  0  0  0  0  0  0  0  0  0  0  0
    ##         3   1  0 11  4  6  5  2  1  0  0  2  1  1  1  2  0  0  0  0  0  0
    ##         4   0  1  2 29 12  7  3  5  0  2  1  0  0  0  0  0  0  0  0  0  0
    ##         5   1  0  2  4 50 11  7  4  5  1  4  2  3  3  1  1  0  0  0  0  0
    ##         6   1  0  5  4  5 65 19 16 13  5  7  7  7  4  4  6  0  0  0  0  0
    ##         7   0  0  1  1  7 18 52 19  9  7  4  8  5  3  4  2  2  0  0  0  0
    ##         8   0  0  1  1  4 22 19 77 33 12  9  2  3  4  2  4  2  0  0  0  0
    ##         9   0  0  2  1  5 15  6 30 76 25  4  8  6  5  3  5  1  0  0  0  0
    ##         10  1  0  2  2  4  8  6  7 17 51 18  8  5  5  2  1  1  1  0  0  0
    ##         11  1  0  1  0  2  5  3  2  1 13 35 11  7  7  7  2  1  2  0  0  0
    ##         12  1  0  1  0  0  3  3  4  2  3 12 24 11  6  4  5  2  0  0  0  0
    ##         13  0  0  0  1  1  5  3  2  5  4  6  3 15 13  4  2  2  0  0  0  0
    ##         14  0  1  0  0  1  3  3  4  1  6  5  6  9 30  9  6  2  2  0  2  0
    ##         15  1  1  1  0  0  3  0  1  5  5  1  1  2  6 12  2  2  1  0  0  0
    ##         16  0  0  0  0  0  2  2  4  1  2  1  4  3  1  7 19  5  1  0  0  0
    ##         17  0  0  0  0  0  0  0  1  0  3  1  0  1  2  3  2  7  1  1  1  0
    ##         18  0  0  0  0  1  0  0  0  0  1  0  0  0  1  0  1  2  2  0  0  1
    ##         19  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0
    ##         20  0  0  0  0  0  1  0  0  0  1  1  0  0  0  0  0  0  1  0  0  0
    ##         21  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.3717          
    ##                  95% CI : (0.3475, 0.3963)
    ##     No Information Rate : 0.1157          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.318           
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity          0.375000 0.764706 0.354839  0.60417  0.50505  0.37143
    ## Specificity          0.991509 0.993464 0.982850  0.97799  0.96616  0.92493
    ## Pos Pred Value       0.315789 0.565217 0.297297  0.46774  0.50505  0.38690
    ## Neg Pred Value       0.993455 0.997375 0.986755  0.98721  0.96616  0.92023
    ## Prevalence           0.010343 0.010989 0.020039  0.03103  0.06399  0.11312
    ## Detection Rate       0.003878 0.008403 0.007111  0.01875  0.03232  0.04202
    ## Detection Prevalence 0.012282 0.014867 0.023917  0.04008  0.06399  0.10860
    ## Balanced Accuracy    0.683254 0.879085 0.668844  0.79108  0.73561  0.64818
    ##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11 Class: 12
    ## Sensitivity           0.39098  0.43017  0.44706   0.35664   0.31532   0.28235
    ## Specificity           0.93635  0.91374  0.91576   0.93732   0.95474   0.96101
    ## Pos Pred Value        0.36620  0.39487  0.39583   0.36691   0.35000   0.29630
    ## Neg Pred Value        0.94235  0.92456  0.93063   0.93466   0.94748   0.95839
    ## Prevalence            0.08597  0.11571  0.10989   0.09244   0.07175   0.05495
    ## Detection Rate        0.03361  0.04977  0.04913   0.03297   0.02262   0.01551
    ## Detection Prevalence  0.09179  0.12605  0.12411   0.08985   0.06464   0.05236
    ## Balanced Accuracy     0.66366  0.67196  0.68141   0.64698   0.63503   0.62168
    ##                      Class: 13 Class: 14 Class: 15 Class: 16 Class: 17
    ## Sensitivity           0.189873   0.32609  0.184615   0.32759  0.241379
    ## Specificity           0.965259   0.95876  0.978408   0.97784  0.989460
    ## Pos Pred Value        0.227273   0.33333  0.272727   0.36538  0.304348
    ## Neg Pred Value        0.956786   0.95745  0.964737   0.97391  0.985564
    ## Prevalence            0.051067   0.05947  0.042017   0.03749  0.018746
    ## Detection Rate        0.009696   0.01939  0.007757   0.01228  0.004525
    ## Detection Prevalence  0.042663   0.05818  0.028442   0.03361  0.014867
    ## Balanced Accuracy     0.577566   0.64242  0.581511   0.65271  0.615420
    ##                      Class: 18 Class: 19 Class: 20 Class: 21
    ## Sensitivity           0.181818 0.0000000  0.000000 0.5000000
    ## Specificity           0.995443 0.9993532  0.997409 1.0000000
    ## Pos Pred Value        0.222222 0.0000000  0.000000 1.0000000
    ## Neg Pred Value        0.994148 0.9993532  0.998056 0.9993532
    ## Prevalence            0.007111 0.0006464  0.001939 0.0012928
    ## Detection Rate        0.001293 0.0000000  0.000000 0.0006464
    ## Detection Prevalence  0.005818 0.0006464  0.002586 0.0006464
    ## Balanced Accuracy     0.588630 0.4996766  0.498705 0.7500000
