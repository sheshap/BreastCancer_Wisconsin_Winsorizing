library(caret)
library(psych)  
library(nnet)   # neural network
library(plyr)

con <- file("breast-cancer-wisconsin.data", "r")
cols <- c("id_number","Clump_Thickness","U_Cell_Size","U_Cell_Shape","Marginal_Adhesion","SE_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses","Class")
instances <- length(readLines("breast-cancer-wisconsin.data"))
df <- data.frame(matrix(ncol = length(cols), nrow = instances))
colnames(df) <- cols
nr<-1
while ( nr<=instances ) {
  line = readLines(con, n = 1)
  if ( length(line) == 0 ) {
    break
  }
  x <- unlist((strsplit(line,",")))
  for(i in 1:length(cols)) {
    df[nr, cols[i]] <- as.numeric(x[i])
  }
  nr=nr+1
}
close(con)
df$id_number<-NULL
save(df, file = "file.RData")

partition_dataset <- function(data, name) {
  inTrain <- createDataPartition(y=data$Class, p=0.7, list=FALSE)
  train <- data[inTrain, ]
  test <- data[-inTrain, ]
  train_file <- paste('./', name, '_train.RData', sep="")
  test_file <- paste('./', name, '_test.RData', sep="")
  save(train, file=train_file)
  save(test, file=test_file)
}

partition_dataset(df, "cancer")

cnn_ret <- NULL    # neural network

run <- function() {
  for(k in 1:15){
    complete_analysis(k)
  }
  Acc<-accuracy(cnn_ret)
  fname <- paste('./allfeatures_winsor.png')
  png(filenam=fname)
  plot(Acc, type = "o", col = "blue", xlab = "Percentage of winsorizing", ylab = "Accuracies", main=paste("All Features Winsorized"))
  dev.off()
}

loadRData <- function(fileName){
  load(fileName)
  get(ls()[ls() != "fileName"])
}
complete_analysis <- function(k) {
  train_file <- paste('./cancer_train.RData', sep="")
  test_file <- paste('./cancer_test.RData', sep="")
  f <- function() .GlobalEnv$train <- 7
  f()
  g <- function() .GlobalEnv$test <- 7
  g()
  train <<- loadRData(train_file)
  test <<- loadRData(test_file)

  for(j in 1:9){
    train[j] <- winsor(train[j], trim = 0.01*k, na.rm=TRUE)
    test[j] <- winsor(test[j], trim = 0.01*k, na.rm=TRUE)
  }
  
  train$Class <- as.factor(train$Class)
  model <- nnet(Class ~ ., data=train, size=1, rang=0.1, decay=5e-4, maxit=1000, trace=FALSE)
  print(model)
  cross_val <- predict(model, test,type="class")
  cross_val <- as.factor(cross_val)
  cnn_ret <<- c(cnn_ret, confusionMatrix(cross_val, test$Class))
  save(cnn_ret, file="./cnn_ret.RData")
}

accuracy <- function(cnn_ret){
    j<-0
    A <- 0
    print("Accuracy")
    while(j<15){
      #cat(sprintf("\"%i\"  \"%f\"\n",as.integer(j+1),as.double(cnn_ret[3+(6*j)]$overall[1])))
      A[j] <- as.double(cnn_ret[3+(6*j)]$overall[1])
      j<-j+1
    }
    plot(A, type="o", col="blue")
    return(A)
}

run()