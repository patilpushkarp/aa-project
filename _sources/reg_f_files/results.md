# Regression Results

Every algorithm has its own pros and cons depending upon the underlying
assumptions and statistical methods used. Thus, to summarize

``` r
df <- read.csv("./../data/regression_data/output/result.csv")
df
```

    ##                     Algorithm      RMSE       MAE
    ## 1           Linear Regression 12.697598 10.429024
    ## 2              kNN Regression  9.851694  7.554406
    ## 3    Decision Tree Regression 10.094446  7.985458
    ## 4     Regression with Bagging  9.160999  7.144364
    ## 5    Random Forest Regression  8.042740  6.107289
    ## 6 Gradient Descent Regression  8.135001  6.217862

For this case, Random Forest algorithm has given the best performance.
