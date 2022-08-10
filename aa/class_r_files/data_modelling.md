# Data Modelling

Since the data has been cleaned, it can now be used to create the
models.

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

Since not many outliers are removed, there is very little information
loss and hence the data without outliers can be used.

``` r
# Load the data
df <- read.csv("./../data/classification_data/intermediates/preprocessed_data_without_outliers.csv")
```

## Partition the data

To evaluate the model, there should be a set of which the model has not
seen and for which the labels are known. Hence, it is necessary to split
the data into training and testing set.

``` r
# Partitioning the data
partition = createDataPartition(df$Rating, p=0.8, list = FALSE)
train = df[partition,]
test = df[-partition,]
```

``` r
# Save the training and testing data
# write.csv(train, "./../data/classification_data/intermediates/train.csv", row.names = TRUE)
# write.csv(test, "./../data/classification_data/intermediates/test.csv", row.names = TRUE)
```

## Algorithms

Following are the algorithms that will be used to model the data:

1.  Multinomial Logistic Regression
2.  Support Vector Machine
3.  Naive Bayes
4.  kNN
5.  Decision Tree
6.  Bagging
7.  Random Forest
8.  Gradient Boosting
