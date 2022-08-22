# Gradient Boosting

``` r
# Load libraries
library(gbm)
```

    ## Loaded gbm 2.1.8

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(knitr)

# Load helpers
source("./../helpers/helper.R")
```

## Import Data

``` r
# Read training and testing data
train <- read.csv("./../data/classification_data/intermediates/train.csv")
test <- read.csv("./../data/classification_data/intermediates/test.csv")
```

## Model Training

``` r
# Train the model
gbm.model = gbm(as.factor(Rating)~., data=train, distribution = "multinomial")
```

    ## Warning: Setting `distribution = "multinomial"` is ill-advised as it is
    ## currently broken. It exists only for backwards compatibility. Use at your own
    ## risk.

``` r
gbm.model
```

    ## gbm(formula = as.factor(Rating) ~ ., distribution = "multinomial", 
    ##     data = train)
    ## A gradient boosted model with multinomial loss function.
    ## 100 iterations were performed.
    ## There were 31 predictors of which 30 had non-zero influence.

``` r
# Predict the samples from test data using the model
result <- predict(gbm.model, test)
```

    ## Using 100 trees...

``` r
# Find the classes from the prediction probabilities
final.result = colnames(result)[apply(result,1,which.max)]

# Print the Confusion matrix
confusion.matrix <- confusionMatrix(as.factor(final.result), as.factor(test$Rating))
```

    ## Warning in levels(reference) != levels(data): longer object length is not a
    ## multiple of shorter object length

    ## Warning in confusionMatrix.default(as.factor(final.result),
    ## as.factor(test$Rating)): Levels are not in the same order for reference and
    ## data. Refactoring data to match.

``` r
plot.custom.confusion.matrix(confusion.matrix$table)
```

![](class_gradient_boosting_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
# Print the accuracy stats of the model
kable(data.frame(confusion.matrix$overall))
```

|                | confusion.matrix.overall |
|:---------------|-------------------------:|
| Accuracy       |                0.3089851 |
| Kappa          |                0.2462804 |
| AccuracyLower  |                0.2860167 |
| AccuracyUpper  |                0.3326809 |
| AccuracyNull   |                0.1215255 |
| AccuracyPValue |                0.0000000 |
| McnemarPValue  |                      NaN |

``` r
# Print validation stats of the model
kable(data.frame(confusion.matrix$byClass))
```

|           | Sensitivity | Specificity | Pos.Pred.Value | Neg.Pred.Value | Precision |    Recall |        F1 | Prevalence | Detection.Rate | Detection.Prevalence | Balanced.Accuracy |
|:----------|------------:|------------:|---------------:|---------------:|----------:|----------:|----------:|-----------:|---------------:|---------------------:|------------------:|
| Class: 1  |   0.2222222 |   0.9947678 |      0.3333333 |      0.9908795 | 0.3333333 | 0.2222222 | 0.2666667 |  0.0116354 |      0.0025856 |            0.0077569 |         0.6084950 |
| Class: 2  |   0.6000000 |   0.9980418 |      0.7500000 |      0.9960912 | 0.7500000 | 0.6000000 | 0.6666667 |  0.0096962 |      0.0058177 |            0.0077569 |         0.7990209 |
| Class: 3  |   0.0238095 |   0.9933555 |      0.0909091 |      0.9733073 | 0.0909091 | 0.0238095 | 0.0377358 |  0.0271493 |      0.0006464 |            0.0071105 |         0.5085825 |
| Class: 4  |   0.0769231 |   0.9926421 |      0.2666667 |      0.9686684 | 0.2666667 | 0.0769231 | 0.1194030 |  0.0336134 |      0.0025856 |            0.0096962 |         0.5347826 |
| Class: 5  |   0.4230769 |   0.9503063 |      0.3113208 |      0.9687717 | 0.3113208 | 0.4230769 | 0.3586957 |  0.0504202 |      0.0213316 |            0.0685197 |         0.6866916 |
| Class: 6  |   0.3372781 |   0.9005806 |      0.2938144 |      0.9172210 | 0.2938144 | 0.3372781 | 0.3140496 |  0.1092437 |      0.0368455 |            0.1254040 |         0.6189293 |
| Class: 7  |   0.2413793 |   0.9493581 |      0.3301887 |      0.9236641 | 0.3301887 | 0.2413793 | 0.2788845 |  0.0937298 |      0.0226244 |            0.0685197 |         0.5953687 |
| Class: 8  |   0.3043478 |   0.9033189 |      0.2677596 |      0.9178886 | 0.2677596 | 0.3043478 | 0.2848837 |  0.1040724 |      0.0316742 |            0.1182935 |         0.6038334 |
| Class: 9  |   0.4095745 |   0.8668138 |      0.2984496 |      0.9138867 | 0.2984496 | 0.4095745 | 0.3452915 |  0.1215255 |      0.0497738 |            0.1667744 |         0.6381942 |
| Class: 10 |   0.2847222 |   0.9465431 |      0.3534483 |      0.9280224 | 0.3534483 | 0.2847222 | 0.3153846 |  0.0930834 |      0.0265029 |            0.0749838 |         0.6156327 |
| Class: 11 |   0.7090909 |   0.9227557 |      0.4126984 |      0.9764359 | 0.4126984 | 0.7090909 | 0.5217391 |  0.0711054 |      0.0504202 |            0.1221719 |         0.8159233 |
| Class: 12 |   0.3058824 |   0.9377565 |      0.2222222 |      0.9587413 | 0.2222222 | 0.3058824 | 0.2574257 |  0.0549451 |      0.0168067 |            0.0756303 |         0.6218194 |
| Class: 13 |   0.2417582 |   0.9759615 |      0.3859649 |      0.9536913 | 0.3859649 | 0.2417582 | 0.2972973 |  0.0588235 |      0.0142211 |            0.0368455 |         0.6088599 |
| Class: 14 |   0.2613636 |   0.9705278 |      0.3484848 |      0.9561107 | 0.3484848 | 0.2613636 | 0.2987013 |  0.0568843 |      0.0148675 |            0.0426632 |         0.6159457 |
| Class: 15 |   0.1568627 |   0.9779412 |      0.1951220 |      0.9714475 | 0.1951220 | 0.1568627 | 0.1739130 |  0.0329670 |      0.0051713 |            0.0265029 |         0.5674020 |
| Class: 16 |   0.1521739 |   0.9713524 |      0.1400000 |      0.9739479 | 0.1400000 | 0.1521739 | 0.1458333 |  0.0297350 |      0.0045249 |            0.0323206 |         0.5617632 |
| Class: 17 |   0.0250000 |   0.9993364 |      0.5000000 |      0.9747573 | 0.5000000 | 0.0250000 | 0.0476190 |  0.0258565 |      0.0006464 |            0.0012928 |         0.5121682 |
| Class: 18 |   0.2000000 |   0.9947781 |      0.2727273 |      0.9921875 | 0.2727273 | 0.2000000 | 0.2307692 |  0.0096962 |      0.0019392 |            0.0071105 |         0.5973890 |
| Class: 19 |   0.0000000 |   0.9993532 |      0.0000000 |      0.9993532 | 0.0000000 | 0.0000000 |       NaN |  0.0006464 |      0.0000000 |            0.0006464 |         0.4996766 |
| Class: 20 |   0.0000000 |   1.0000000 |            NaN |      0.9974144 |        NA | 0.0000000 |        NA |  0.0025856 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 21 |   0.0000000 |   1.0000000 |            NaN |      0.9974144 |        NA | 0.0000000 |        NA |  0.0025856 |      0.0000000 |            0.0000000 |         0.5000000 |

``` r
# Save the results
algorithm <- "Gradient.Boosting"
save.class.acc.result(confusion.matrix$overall, algorithm)
save.class.pvv.result(confusion.matrix$byClass, algorithm)
```
