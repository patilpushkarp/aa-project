# Data Modelling

Since the data has been cleaned, it can now be used to create the
models.

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

Let the start modelling with the data from which outliers are removed.

``` r
# Load the data
df <- read.csv("./../data/regression_data/intermediates/preprocessed_data_without_outliers.csv")
```

``` r
# Get the dimensions of data
dim(df)
```

    ## [1] 1121   31

## Partition the data

To evaluate the model, there should be a set of which the model has not
seen and for which the labels are known. Hence, it is necessary to split
the data into training and testing set.

``` r
# Partitioning the data
?write.csv
partition = createDataPartition(df$CA, p=0.8, list = FALSE)
train = df[partition,]
test = df[-partition,]
```

``` r
names(train)
```

    ##  [1] "Apps"                 "Mins"                 "Mins.Gm"             
    ##  [4] "Height"               "Weight"               "Age"                 
    ##  [7] "Av.Rat"               "Gls"                  "Gls.90"              
    ## [10] "Shot.."               "xG"                   "Ch.C.90"             
    ## [13] "Asts.90"              "K.Ps.90"              "Pas.."               
    ## [16] "Cr.C.A"               "Drb.90"               "Distance"            
    ## [19] "Hdr.."                "K.Tck"                "Fls"                 
    ## [22] "PoM"                  "Aer.A.90"             "Off"                 
    ## [25] "Tck.R"                "CA"                   "Gls.xG"              
    ## [28] "Dist.Mins"            "Value"                "Lower.Transfer.Value"
    ## [31] "Upper.Transfer.Value"

``` r
# Save the training and testing data
write.csv(train, "./../data/regression_data/intermediates/train.csv", row.names = FALSE)
write.csv(test, "./../data/regression_data/intermediates/test.csv", row.names = FALSE)
```

## Regression Algorithms

Following are the classification algorithms that will be used to model
the data:

1.  Linear Regression
2.  kNN
3.  Decision Tree
4.  Bagging
5.  Random Forest
6.  Gradient Boosting
