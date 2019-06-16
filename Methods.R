## Author: Robert Edwards
## University of Glasgow
## Multivariate Methods 2018
## All Classifying Methods

library(kknn)

#############################################
### Linear Regression
#############################################
classify.lr <- function(train.data, valid.data) {
  
  # Change Class labels to numeric 
  train.data <- as.data.frame(cbind(as.numeric(train.data[,1]), train.data[,-1]))
  valid.data <- as.data.frame(cbind(as.numeric(valid.data[,1]), valid.data[,-1]))
  
  print("as numeric train data")
  print(table(as.numeric(train.data[,1])))
  print(table(as.numeric(train.data[,1]))[1])
  colnames(train.data) <- c("Class", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10")
  colnames(valid.data) <- c("Class", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10")
  

  ## Fitting the linear model to the training data.
  res <- lm(Class~.,data=train.data)
  
  ## Using the predict function to predict for the validation data.
  pred.valid <- predict(res, valid.data[, -1])

  ## Changing the predictions to predited labels.
  pred.valid.label <- round(pred.valid)
  
  pred.valid.label[pred.valid.label < 1] <- 1
  pred.valid.label[pred.valid.label > 12] <- 12 
  

  ## Cross-classification table.
  xtab <- table(valid.data[, 1], pred.valid.label)

  ## Total Correctly Classiffied
  corr.class.total <- sum(pred.valid.label == valid.data[, 1])
  ## Correct classification rate (CCR).
  corr.class.rate <- sum(pred.valid.label == valid.data[, 1]) / length(pred.valid.label)
  ## Class-specific CCRs.
  ccr.class <- diag(xtab) / rowSums(xtab)
  
  lr.err <- 1 - sum(diag(xtab)) / sum(xtab)
  
  errors <- list(lr.err, 1-ccr.class)
  return(errors)
} # End classify.lr


#############################################
### KNN
#############################################
classify.knn <- function(train.data, test.data, k=1, dist=2, ker="rectangular") {
  
  n.test <- nrow(test.data)

  k.nn.tr <- kknn(formula=formula(Class~.), train=train.data, test=train.data[-1], k=k, distance=dist, kernel=ker)
  k.nn.tst <- kknn(formula=formula(Class~.), train=train.data, test=test.data[-1], k=k, distance=dist, kernel=ker)
  
  ## Class-specific CCRs.
  xtab <- table(k.nn.tst$fitted.values, test.data[, 1]) 
  ccr.class <- diag(xtab) / rowSums(xtab)
  ## Calculate and store the test set correct classification rate
  knn.err <- sum(k.nn.tst$fitted.values == test.data[, 1]) / n.test

  errors <- list(knn.err, 1-ccr.class)
  return(errors)
} # end KNN



#############################################
### CVA
#############################################
classify.cva <- function(train.data, valid.data) {
  ## Set Prior probabilites equal
  data.lda <- lda(train.data[, -1], train.data[, 1], prior=rep(1/nlevels(train.data[, 1]), 12) )


  ## predict the labels based on the model.
  data.valid.ld <- predict(data.lda, newdata=valid.data[, -1]) 
  
  ## cross-classification table 
  xtab <- table(valid.data[, 1], data.valid.ld$class) 

  ## Total Correctly Classiffied
  corr.class.total <- sum(diag(xtab))
  ## Correct Classification Rate
  corr.class.rate <- sum(diag(xtab)) / sum(xtab)
  ## Misclassification rate =  1 - correct classifcation rate
  mcr <- 1 - sum(diag(xtab)) / sum(xtab)

  ## Class-specific CCRs.
  ccr.class <- diag(xtab) / rowSums(xtab)

  ## Error Rate
  cva.err <- 1 - sum(diag(xtab)) / sum(xtab)
  
  errors <- list(cva.err, 1-ccr.class)
  return(errors)
} #end classify.cva



#############################################
## LDA
#############################################
classify.lda <- function(train.data, valid.data) {
  
  ## Check if n > 0 and remove those classes
  if ( sum(table(train.data$Class) < 1 ) != 0) {
    too.few <- levels(train.data$Class)[table(train.data$Class) < 1]
    cat("Removing from train data: ", too.few, "\n")
    
    for (i in 1:length(too.few)) {
      train.data <- subset(train.data, Class != too.few[i]) # if n < 0 for each class
      valid.data <- subset(valid.data, Class != too.few[i]) # if n < 0 for each class
    }
    
    train.data$Class <- factor(train.data$Class)
    valid.data$Class <- factor(valid.data$Class)
  }
  if ( sum(table(valid.data$Class) < 1 ) != 0) {
    too.few <- levels(valid.data$Class)[table(valid.data$Class) < 1]
    cat("Removing from valid data: ", too.few, "\n")
    
    for (i in 1:length(too.few)) {
      train.data <- subset(train.data, Class != too.few[i]) # if n < 0 for each class
      valid.data <- subset(valid.data, Class != too.few[i]) # if n < 0 for each class
    }
    train.data$Class <- factor(train.data$Class)
    valid.data$Class <- factor(valid.data$Class)
  }
  

  ## Let lda set priors
  data.lda <- lda(Class~., data=train.data)
  
  ## Predicting the classes on the data.
  data.valid.pred.lda<- predict(data.lda, valid.data)
  
  ## Constructing a cross-classification table of the real versus predicted classifications.
  xtab<-table(valid.data$Class, data.valid.pred.lda$class)

  ## Class-specific CCRs.
  ccr.class <- diag(xtab) / rowSums(xtab)
  
  ## Calculating the misclassification rate.
  lda.err <- 1 - sum(diag(xtab)) / sum(xtab)
  
  errors <- list(lda.err, 1-ccr.class)
  return(errors)
} # end classify.lda


############################################
## QDA
############################################
classify.qda <- function(train.data, valid.data) {
    
  ## Check if n < p and remove those classes
  if ( sum(table(train.data$Class) <= ncol(train.data[-1]) ) != 0) {
    too.few.train <- levels(train.data$Class)[table(train.data$Class) <= ncol(train.data[-1])]
    cat("Removing from train data: ", too.few.train, "\n")
  }
  if ( sum(table(valid.data$Class) <= ncol(valid.data[-1]) ) != 0) {
    too.few.valid <- levels(valid.data$Class)[table(valid.data$Class) <= ncol(valid.data[-1])]
    cat("Removing from train data: ", too.few.valid, "\n")
  }
  
  too.few <- c(too.few.train, too.few.valid)
  
  if ( length(too.few) != 0) {
    for (i in 1:length(too.few)) {
      train.data <- subset(train.data, Class != too.few[i]) # if n < p for each class
      valid.data <- subset(valid.data, Class != too.few[i]) # if n < p for each class
    }
    train.data$Class <- factor(train.data$Class)
    valid.data$Class <- factor(valid.data$Class)
  }
    
  data.qda <- qda(Class~., data=train.data)
  
  ## Prediction 
  data.valid.pred.qda <- predict(data.qda, valid.data)
  
  ## Constructing a cross-classification table of the real versus predicted classifications.
  xtab <- table(valid.data$Class, data.valid.pred.qda$class)

  ## Class-specific CCRs.
  ccr.class <- diag(xtab) / rowSums(xtab)

  ## Calculating the error rate.
  qda.err <- 1 - sum(diag(xtab)) / sum(xtab)
    
  errors <- list(qda.err, 1-ccr.class)
  return(errors)
} # end classify.qda


###############################################
## TREES
###############################################
classify.tree <- function(train.data, valid.data) {
  
  set.seed(126)
  library(rpart)
  ## Set"class" to ensure we get a classification 
  data.rp <- rpart(Class~. , data=train.data, method="class")
  
  ## We predict the classifications based on the fitted tree using the predict command with the argument type set to "class", otherwise it returns a matrix of posterior probabilities of each obseration belonging to each class.
  data.valid.rp.pred <- predict(data.rp, valid.data, type="class")
  
  ## Constructing a cross-classification table of the real versus predicted classifications.
  xtab <- table(valid.data$Class, data.valid.rp.pred)

  ## Taking a look at the posterior probabilities for the observations we got the prediction wrong for.
  data.valid.rp.prob <- predict(data.rp, valid.data)
  data.valid.rp.prob[valid.data$Class != data.valid.rp.pred, ]
  
  ## Class-specific CCRs.
  ccr.class <- diag(xtab) / rowSums(xtab)
  ## Calculating the misclassification rate.
  tree.err <- 1 - sum(diag(xtab)) / sum(xtab)
  
  errors <- list(tree.err, 1-ccr.class)
  return(errors)
} # end classify.tree



###############################################
## RANDOM FORESTS
###############################################
classify.rf <- function(train.data, valid.data) {
  library(randomForest)
  set.seed(126)
  
  too.few <- NA
  ## Remove empty classes
  if ( sum(table(train.data$Class) < 1 ) > 0) {
    too.few <- levels(train.data$Class)[table(train.data$Class) < 1]
    cat("Removing from train data: ", too.few, "\n")

    for (i in 1:length(too.few)) {
      train.data <- subset(train.data, Class != too.few[i]) # if n < p for each class
      valid.data <- subset(valid.data, Class != too.few[i]) # if n < p for each class
    }
    train.data$Class <- factor(train.data$Class)
    valid.data$Class <- factor(valid.data$Class)
  }
  if ( sum(table(valid.data$Class) < 1 ) > 0) {
    too.few <- NA
    too.few <- levels(valid.data$Class)[table(valid.data$Class) < 1]
    cat("Removing from valid data: ", too.few, "\n")
    
    for (i in 1:length(too.few)) {
      train.data <- subset(train.data, Class != too.few[i]) # if n < p for each class
      valid.data <- subset(valid.data, Class != too.few[i]) # if n < p for each class
    }
    train.data$Class <- factor(train.data$Class)
    valid.data$Class <- factor(valid.data$Class)
  }
  
  
  data.rf <- randomForest(Class~., data=train.data) 
  
  ## Prediction
  data.valid.rf.pred <- predict(data.rf, newdata=valid.data)
  
  ## Cross classification
  xtab <- table(valid.data$Class, data.valid.rf.pred)
  
  ## Class-specific CCRs.
  ccr.class <- diag(xtab) / rowSums(xtab)
  ## Calculating the misclassification rate.
  rf.err <- 1 - sum(diag(xtab)) / sum(xtab)
    
  errors <- list(rf.err, 1-ccr.class)
  return(errors)
} # end classify.rf

