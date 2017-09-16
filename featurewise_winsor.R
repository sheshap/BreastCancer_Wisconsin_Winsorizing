# This code is developed by Shivanand Venkanna Sheshappanavar for the below course requirement
# Course: CIS 731 Artificial Neural Network
# Professor: Dr. Chilkuri Mohan
# Homework Assignment: 1 (Effect of Winsorizing of data on accuracies)
#Open this file in RStudio, Set working directory to the directory in which the script is store along with data file and select entire code and click run.
library(devtools)
library(caret)
library(psych)  
library(nnet)   # neural network
library(plyr)

#open data file and import data by reading line by line
file_pointer <- file("breast-cancer-wisconsin.data", "r")
instances <- length(readLines("breast-cancer-wisconsin.data"))

#assign column names as mentioned in the dataset description
cols <- c("id_number","Clump_Thickness","U_Cell_Size","U_Cell_Shape","Marginal_Adhesion","SE_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses","Class")

#create a data frame of the size of the number of instance * columns
df <- data.frame(matrix(ncol = length(cols), nrow = instances))
colnames(df) <- cols
nr<-1

#Read data line by line from the dataset file
while ( nr<=instances ) {
  line = readLines(file_pointer, n = 1)
  if ( length(line) == 0 ) {
    break
  }
  x <- unlist((strsplit(line,",")))
  for(i in 1:length(cols)) {
    df[nr, cols[i]] <- as.numeric(x[i])
  }
  nr=nr+1
}
close(file_pointer)

#ignore id_number feature by setting it the column to NULL
df$id_number<-NULL
save(df, file = "file.RData")

#partition dataset into training and testing with 70% and 30% of the instances randomly distributed among the two respectively
partition_dataset <- function(data, name) {
  inTrain <- createDataPartition(y=data$Class, p=0.7, list=FALSE)
  train <- data[inTrain, ]
  test <- data[-inTrain, ]
  #saves partitioned data into train and test files 
  train_file <- paste('./', name, '_train.RData', sep="")
  test_file <- paste('./', name, '_test.RData', sep="")
  save(train, file=train_file)
  save(test, file=test_file)
}
#invoke partiotion function during runtime
partition_dataset(df, "cancer")

nn_ret <- NULL    # neural network
Acc <- 0          #an array to store 
niterations <- 15 # up to 15 percentage of winsorizing
nfeatures <- 9 #9 significant features
nres <- niterations*nfeatures  #minimum resulting number of accuracies

#function to invoke the training,testing and analysis
run <- function() {
 for(p in 1:niterations){
  for(j in 1:nfeatures){
       analysis(p,j)
  }
 }
  Acc<-accuracy(nn_ret,nres)
  #plots each percentage winsorization
  plot_acc_win(Acc)
  #plots each feature winsorization
  plot_acc_feature(Acc)
}

#plots each feature winsorization
plot_acc_feature<-function(Acc){
  i<-1
  j<-1
  tmp <- matrix(Acc, nrow=9, ncol=15)
  newAcc <- as.vector(t(tmp))
  while(i< length(Acc)){
    #saves results in Feature directory
    fname <- paste('./Feature/plot',j,'.png',sep="")
    temp<-newAcc[i:(i+niterations-1)]
    png(filenam=fname)
    plot(temp, type = "o", col = "red", xlab = "Percentage of winsorizing", ylab = "Accuracies", main=paste("Feature",j,"-", cols[j+1],"winsorizing"))
    dev.off()
    i<-i+niterations
    j<-j+1
  }
}

#plots each percentage winsorization
plot_acc_win <- function(Acc){
  i <- 1
  j<-1
  while(i< length(Acc)){
    #saves results in Percentage directory
    pname <- paste('./Percentage/plot',j,'.png',sep="")
    temp<-Acc[i:(i+nfeatures-1)]
    png(filenam=pname)
    plot(temp, type = "o", col = "green", xlab = "Feature Numbers", ylab = "Accuracies", main=paste("All features winsorizing percentage:",j))
    dev.off()
    i<-i+nfeatures
    j<-j+1
  }
}

#loads the datafile
loadRData <- function(fileName){
  load(fileName)
  get(ls()[ls() != "fileName"])
}

#picks train and test files, modifies train file by winsorizing 
analysis <- function(p,j) {
  train_file <- paste('./cancer_train.RData', sep="")
  test_file <- paste('./cancer_test.RData', sep="")
  f <- function() .GlobalEnv$train <- 7
  f()
  g <- function() .GlobalEnv$test <- 7
  g()
  train <<- loadRData(train_file)
  test <<- loadRData(test_file)
  train[j] <- winsor(train[j], trim = 0.01*p, na.rm=TRUE)
  test[j] <- winsor(test[j], trim = 0.01*p, na.rm=TRUE)
  neural_network(train,test)
}

#single node neural network invoked to build a model and the model is tested with test data
neural_network <- function(train,test) {
  train$Class <- as.factor(train$Class)
  model <- nnet(Class ~ ., data=train, size=1, rang=0.1, decay=5e-4, maxit=1000, trace=FALSE)
  cross_val <- predict(model, test,type="class")
  cross_val <- as.factor(cross_val)
  nn_ret <<- c(nn_ret, confusionMatrix(cross_val, test$Class))
  #saves confusion matrix to the below file
  save(nn_ret, file="./nn_ret.RData")
}

#extracts 
accuracy <- function(nn_ret, nres){
  i<-0
  A<-1
  while(i<nres){
    if (i %% nfeatures == 0){
      #cat(sprintf("=====\"%i\"====\n",as.integer(i/nfeatures)+1))
    }
    p=as.integer(i+1)%%nfeatures
    if (p == 0){
      p=nfeatures
    }
    #cat(sprintf("\"%i\"  \"%f\"\n",p,as.double(nn_ret[3+(6*i)]$overall[1])))
    A[i+1] <- as.double(nn_ret[3+(6*i)]$overall[1])
    i<-i+1
  }
  return(A)
}
#final step which will invoke the loaded code
run()