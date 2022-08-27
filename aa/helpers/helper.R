library(cvms)
library(tibble)

save.reg.result <- function(RMSE=0, MAE=0, algo.name=NULL){
  reg.file.name <- "./../data/regression_data/output/result.csv"
  if (file.exists(reg.file.name)){
    df <- read.csv(reg.file.name)
    if (any(df$Algorithm == algo.name)){
      df[which(df$Algorithm == algo.name), "RMSE"] = RMSE
      df[which(df$Algorithm == algo.name), "MAE"] = MAE
    }else{
      df[nrow(df) + 1,] = c(algo.name, RMSE, MAE)
      write.csv(df, reg.file.name, row.names = FALSE)
    }
  }else{
    df <- data.frame(Algorithm=c(algo.name), RMSE=c(RMSE), MAE=c(MAE))
    write.csv(df, reg.file.name, row.names = FALSE)
  }
}

save.class.acc.result <- function(overall=NULL, algo.name=NULL){
  class.file.name <- "./../data/classification_data/output/result_acc.csv"
  if (!is.null(overall)){
    accuracy = overall["Accuracy"]
    accuracy.pvalue = overall["AccuracyPValue"]
    kappa = overall["Kappa"]
    if (file.exists(class.file.name)){
      df <- read.csv(class.file.name)
      if (any(df$Algorithm == algo.name)){
        df[which(df$Algorithm == algo.name), "Accuracy"] = accuracy
        df[which(df$Algorithm == algo.name), "P-Value"] = accuracy.pvalue
        df[which(df$Algorithm == algo.name), "Kappa"] = kappa
      }else{
        df[nrow(df) + 1,] = c(algo.name, accuracy, accuracy.pvalue, kappa)
        write.csv(df, class.file.name, row.names = FALSE)
      }
    }else{
      df <- data.frame(Algorithm=c(algo.name), Accuracy=c(accuracy), PValue=c(accuracy.pvalue), Kappa=c(kappa))
      write.csv(df, class.file.name, row.names = FALSE)
    }
  }
}

save.class.pvv.result <- function(result=NULL, algo.name=NULL){
  class.file.name <- "./../data/classification_data/output/result_pvv.csv"
  classes = c("AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C")
  result[is.na(result)] <- 0
  if (!is.null(result)){
    if (file.exists(class.file.name)){
      df <- read.csv(class.file.name)
      df[,algo.name] <- result[, "Pos Pred Value"]
      write.csv(df, class.file.name, row.names = FALSE)
    }else{
      df <- data.frame(column1=classes, column2=result[, "Pos Pred Value"])
      names(df) <- c("Classes", algo.name)
      write.csv(df, class.file.name, row.names = FALSE)
    }
  }
}

plot.custom.confusion.matrix <- function(conf.mat.table){
  cfm <- as_tibble(conf.mat.table)
  class.order <- rev(c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"))
  plot_confusion_matrix(cfm, 
                        target_col = "Reference", 
                        prediction_col = "Prediction",
                        counts_col = "n",
                        font_counts = font(
                          size = 2.5
                        ),
                        add_normalized = FALSE,
                        add_col_percentages = FALSE,
                        add_row_percentages = FALSE,
                        class_order = class.order
                        )
}

