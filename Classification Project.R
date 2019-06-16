## Author: Robert Edwards
## University of Glasgow
## Multivariate Methods 2018
## Classification Project

setwd("~/OneDrive - University of Glasgow/University of Glasgow/Multivariate Methods/Project")
source("Methods.R")
library(class)
library(MASS)
library(e1071)
library(knitr)
library(kableExtra)

# Load data
avila <- Avila.48

## Creating a data frame with the Class outcome and the 12 explanatory variables
auth.lab <- avila[,11]
data.avila <- as.data.frame(cbind(auth.lab, avila[,-11]))
colnames(data.avila) <- c("Class", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10")


############################################
## Inspecting the Data
############################################
colors1 <- c("brown3", "brown3", "brown3", "darkgoldenrod2", "darkgoldenrod2", "darkgoldenrod2", "darkgoldenrod2", "chocolate1", "chocolate1", "chocolate1")
boxplot(data.avila[-1], main="Extracted Features",
        col=colors1,
        pars=list(outcol=colors1))

kable(round(cov(data.avila[-1]), 4), caption="Covariance Matrix") %>%
  kable_styling(bootstrap_options = "striped",
                full_width = F, 
                position = "center")

kable(round(cor(data.avila[-1]), 4), caption="Correlation Matrix") %>%
  kable_styling(bootstrap_options = "striped",
                full_width = F, 
                position = "center")

class.count <- as.data.frame(table(data.avila[1]))
colnames(class.count) <- c("Class", "Frequency")
class.count <- class.count[c("Frequency", "Class")]

kable(t(class.count), caption="Class Frequency") %>%
  kable_styling(bootstrap_options = "striped",
                full_width = F, 
                position = "center")

dddpanel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}

pairs(data.avila[, -1], main="Extracted Features",
      diag.panel=panel.hist,
      pch=20)


############################################
## Splitting the data frame into 3 sets: training, validation and test.
############################################
n <- nrow(avila)
ind1 <- sample(c(1:n), round(n / 2))          # Training set
ind2 <- sample(c(1:n)[-ind1], round(n / 4))   # Validation set
ind3 <- setdiff(c(1:n), c(ind1, ind2))        # Test set
train.avila <- data.avila[ind1, ]
valid.avila <- data.avila[ind2, ]
test.avila <- data.avila[ind3, ]

table(train.avila$Class)
table(valid.avila$Class)
table(test.avila$Class)


############################################
## Optimizing KNN
############################################
rect.err.rates <- as.data.frame(matrix(data=NA, nrow=itr/2, ncol=5))
names(rect.err.rates) <- c("K", "dist=0.1", "dist=1", "dist=2", "dist=100")
tri.err.rates <- as.data.frame(matrix(data=NA, nrow=itr/2, ncol=5))
names(tri.err.rates) <- c("K", "dist=0.1", "dist=1", "dist=2", "dist=100")
gaus.err.rates <- as.data.frame(matrix(data=NA, nrow=itr/2, ncol=5))
names(gaus.err.rates) <- c("K", "dist=0.1", "dist=1", "dist=2", "dist=100")
opt.err.rates <- as.data.frame(matrix(data=NA, nrow=itr/2, ncol=5))
names(opt.err.rates) <- c("K", "dist=0.1", "dist=1", "dist=2", "dist=100")

itr <- 20
for (i in seq(1, itr, 2)) {
  set.seed(i)
  n <- nrow(avila)
  ind1 <- sample(c(1:n), round(n / 2))          # Training set
  ind2 <- sample(c(1:n)[-ind1], round(n / 4))   # Validation set
  ind3 <- setdiff(c(1:n), c(ind1, ind2))        # Test set
  train.avila <- data.avila[ind1, ]
  valid.avila <- data.avila[ind2, ]
  test.avila <- data.avila[ind3, ]
  
  rect.err.rates[ceiling(i/2),1] <- i
  rect.err.rates[ceiling(i/2),2] <- classify.knn(train.avila, valid.avila, k=i, dist=0.1, ker="rectangular") # 
  rect.err.rates[ceiling(i/2),3] <- classify.knn(train.avila, valid.avila, k=i, dist=1, ker="rectangular") # Manhattan Distance
  rect.err.rates[ceiling(i/2),4] <- classify.knn(train.avila, valid.avila, k=i, dist=2, ker="rectangular") # Euclidean Distance
  rect.err.rates[ceiling(i/2),5] <- classify.knn(train.avila, valid.avila, k=i, dist=10, ker="rectangular") # Chebyshev's Distance

  tri.err.rates[ceiling(i/2),1] <- i
  tri.err.rates[ceiling(i/2),2] <- classify.knn(train.avila, valid.avila, k=i, dist=0.1, ker="triangular") # 
  tri.err.rates[ceiling(i/2),3] <- classify.knn(train.avila, valid.avila, k=i, dist=1, ker="triangular") # Manhattan Distance
  tri.err.rates[ceiling(i/2),4] <- classify.knn(train.avila, valid.avila, k=i, dist=2, ker="triangular") # Euclidean Distance
  tri.err.rates[ceiling(i/2),5] <- classify.knn(train.avila, valid.avila, k=i, dist=10, ker="triangular") # Chebyshev's Distance
  
  gaus.err.rates[ceiling(i/2),1] <- i
  gaus.err.rates[ceiling(i/2),2] <- classify.knn(train.avila, valid.avila, k=i, dist=0.1, ker="gaussian") # 
  gaus.err.rates[ceiling(i/2),3] <- classify.knn(train.avila, valid.avila, k=i, dist=1, ker="gaussian") # Manhattan Distance
  gaus.err.rates[ceiling(i/2),4] <- classify.knn(train.avila, valid.avila, k=i, dist=2, ker="gaussian") # Euclidean Distance
  gaus.err.rates[ceiling(i/2),5] <- classify.knn(train.avila, valid.avila, k=i, dist=10, ker="gaussian") # Chebyshev's Distance
  
  opt.err.rates[ceiling(i/2),1] <- i
  opt.err.rates[ceiling(i/2),2] <- classify.knn(train.avila, valid.avila, k=i, dist=0.1, ker="optimal") # 
  opt.err.rates[ceiling(i/2),3] <- classify.knn(train.avila, valid.avila, k=i, dist=1, ker="optimal") # Manhattan
  opt.err.rates[ceiling(i/2),4] <- classify.knn(train.avila, valid.avila, k=i, dist=2, ker="optimal") # Euclidean
  opt.err.rates[ceiling(i/2),5] <- classify.knn(train.avila, valid.avila, k=i, dist=10, ker="optimal") # Chebyshev's
}

par(mfcol=c(2,2), mar=c(4.1, 4.1, 4.1, 2), oma=c(0,0,0,6), xpd=NA)
plot(rect.err.rates[,1], rect.err.rates[,2],
     main="Rectangular",
     xlab="K Values",
     ylab="Error Rate",
     ylim=c(0,1),
     col="red")
points(rect.err.rates[,1], rect.err.rates[,3], col="purple")
points(rect.err.rates[,1], rect.err.rates[,4], col="blue")
points(rect.err.rates[,1], rect.err.rates[,5], col="orange")

plot(tri.err.rates[,1], tri.err.rates[,2],
     main ="Triangular",
     xlab="K Values",
     ylab="Error Rate",
     ylim=c(0,1),
     col="red")
points(tri.err.rates[,1], tri.err.rates[,3],col="purple")
points(tri.err.rates[,1], tri.err.rates[,4],col="blue")
points(tri.err.rates[,1], tri.err.rates[,5],col="orange")

plot(gaus.err.rates[,1], gaus.err.rates[,2],
     main ="Guassian",
     xlab="K Values",
     ylab="Error Rate",
     ylim=c(0,1),
     col="red")
points(gaus.err.rates[,1], gaus.err.rates[,3],col="purple")
points(gaus.err.rates[,1], gaus.err.rates[,4],col="blue")
points(gaus.err.rates[,1], gaus.err.rates[,5],col="orange")

plot(opt.err.rates[,1], opt.err.rates[,2],
     main ="Optimal",
     xlab="K Values",
     ylab="Error Rate",
     ylim=c(0,1),
     col="red")
points(opt.err.rates[,1], opt.err.rates[,3],col="purple")
points(opt.err.rates[,1], opt.err.rates[,4],col="blue")
points(opt.err.rates[,1], opt.err.rates[,5],col="orange")
leg <- c("0.1", "1", "2", "10")
legend("topright", legend = leg, col=c("red","purple","blue","orange"), pch=16, inset=c(-0.75, -2), bty="n")
#legend("topright", legend = leg, col=c("red","purple","blue","orange"),  pch=16, inset=c(-1.3, -4), bty="n")
legend("topright", legend="Minkowski\nDistance", inset=c(-1.15, -3), bty="n")


############################################
## Model Selection
############################################
set.seed(2)
itr=1
err.rates <- as.data.frame(matrix(data=NA, nrow=itr, ncol=8))
names(err.rates) <- c("LR", "KNN", "CVA", "LDA", "QDA", "TREE", "RF")


for (i in 1:itr) {
  n <- nrow(avila)
  ind1 <- sample(c(1:n), round(n / 2))          # Training set
  ind2 <- sample(c(1:n)[-ind1], round(n / 4))   # Validation set
  ind3 <- setdiff(c(1:n), c(ind1, ind2))        # Test set
  train.avila <- data.avila[ind1, ]
  valid.avila <- data.avila[ind2, ]
  test.avila <- data.avila[ind3, ]
  
  mcr <- classify.lr(train.avila, valid.avila)
  err.rates[i,1] <- mcr[1]
  err.rate.lr <- as.data.frame(unlist(mcr[-1]))
  mcr <- classify.knn(train.avila, valid.avila, k=11, dist=10, ker="rectangular")
  err.rates[i,2] <- mcr[1]
  err.rate.knn <- as.data.frame(unlist(mcr[-1]))
  mcr <- classify.cva(train.avila, valid.avila)
  err.rates[i,3] <- mcr[1]
  err.rate.cva <- as.data.frame(unlist(mcr[-1]))
  mcr <- classify.lda(train.avila, valid.avila)
  err.rates[i,4] <- mcr[1]
  err.rate.lda <- as.data.frame(unlist(mcr[-1]))
  mcr <- classify.qda(train.avila, valid.avila)
  err.rates[i,5] <- mcr[1]
  err.rate.qda <- as.data.frame(unlist(mcr[-1]))
  mcr <- classify.tree(train.avila, valid.avila)
  err.rates[i,6] <- mcr[1]
  err.rate.tree <- as.data.frame(unlist(mcr[-1]))
  mcr <- classify.rf(train.avila, valid.avila)
  err.rates[i,7] <- mcr[1]
  err.rate.rf <- as.data.frame(unlist(mcr[-1]))
}

err.rates
boxplot(err.rates[-8], main="Classification Error Rates by Method",
        ylab="Error Rate",
        col=c("firebrick", "darkorchid3", "dodgerblue2", "dodgerblue2", "dodgerblue2", "chartreuse3", "chartreuse3"))

kable(round(err.rates, 4)) %>%
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center")
  #row_spec(0, angle = -45)

#####################################
## Model Selection
####################################
test.err.rates <- as.data.frame(matrix(data=NA, nrow=itr, ncol=7))
names(test.err.rates) <- c("LR", "KNN", "CVA", "LDA", "QDA", "TREE", "RF")
mcr <- classify.lr(train.avila, test.avila)
test.err.rates[1] <- mcr[1]
mcr <- classify.knn(train.avila, test.avila, k=17, dist=10, ker="rectangular")
test.err.rates[2] <- mcr[1]
mcr <- classify.cva(train.avila, test.avila)
test.err.rates[3] <- mcr[1]
mcr <- classify.lda(train.avila, test.avila)
test.err.rates[4] <- mcr[1]
mcr <- classify.qda(train.avila, test.avila)
test.err.rates[5] <- mcr[1]
mcr <- classify.tree(train.avila, test.avila)
test.err.rates[6] <- mcr[1]
mcr <- classify.rf(train.avila, test.avila)
test.err.rates[7] <- mcr[1]
test.err.rates

kable(round(test.err.rates, 4)) %>%
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center")

#####################################
## Class Specific Error Rates
####################################
err.rates.class <- cbind(err.rate.lr,
                         err.rate.knn,
                         err.rate.cva,
                         err.rate.lda,
                         err.rate.qda,
                         err.rate.tree,
                         err.rate.rf)
rownames(err.rates.class) <- c("A","B","C","D","E","F","G","H","I","W","X","Y") 
colnames(err.rates.class) <- c("LR", "KNN", "CVA", "LDA", "QDA", "TREE", "RF")
err.rates.class

kable(round(err.rates.class, 4)) %>%
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center")

par(mfcol=c(1,1), mar=c(4, 4, 4, 4), oma=c(0,0,0,0), xpd=NA)
plot(err.rates.class[,1],
     main ="Error Rates by Class",
     xlab="",
     xaxt='n',
     ylab="Error Rate",
     ylim=c(0,1),
     col="red")
axis(1, at = 1:nrow(err.rates.class), labels = rownames(err.rates.class), las=1)
points(err.rates.class[,2],col="purple")
points(err.rates.class[,3],col="blue")
points(err.rates.class[,4],col="orange")
points(err.rates.class[,5],col="teal")
points(err.rates.class[,6],col="green")
points(err.rates.class[,7],col="brown")

lines(err.rates.class[,1],col="red")
lines(err.rates.class[,2],col="purple")
lines(err.rates.class[,3],col="blue")
lines(err.rates.class[,4],col="orange")
lines(err.rates.class[,5],col="teal")
lines(err.rates.class[,6],col="green")
lines(err.rates.class[,7],col="brown")

lda.err <- rbind(err.rate.lda[1,],
                 "-",
                 err.rate.lda[2,],
                 err.rate.lda[3,],
                 err.rate.lda[4,],
                 err.rate.lda[5,],
                 err.rate.lda[6,],
                 err.rate.lda[7,],
                 err.rate.lda[8,],
                 err.rate.lda[9,],
                 err.rate.lda[10,],
                 err.rate.lda[11,])
rownames(lda.err) <- c("A","B","C","D","E","F","G","H","I","W","X","Y")                 
err.rate.lda <- lda.err

knn.err <- rbind(err.rate.knn[1,],
                 "-",
                 "-",
                 err.rate.knn[4,],
                 err.rate.knn[5,],
                 err.rate.knn[6,],
                 err.rate.knn[7,],
                 err.rate.knn[8,],
                 err.rate.knn[9,],
                 "-",
                 err.rate.knn[11,],
                 err.rate.knn[12,])
rownames(knn.err) <- c("A","B","C","D","E","F","G","H","I","W","X","Y")                 
err.rate.knn <- knn.err

qda.err <- rbind(err.rate.qda[1,],
                 "-",
                 "-",
                 err.rate.qda[2,],
                 err.rate.qda[3,],
                 err.rate.qda[4,],
                 err.rate.qda[5,],
                 err.rate.qda[6,],
                 err.rate.qda[7,],
                 "-",
                 err.rate.qda[8,],
                 err.rate.qda[9,])
rownames(qda.err) <- c("A","B","C","D","E","F","G","H","I","W","X","Y")                 
err.rate.qda <- qda.err

rf.err <- rbind(err.rate.rf[1,],
                 "-",
                 err.rate.rf[2,],
                 err.rate.rf[3,],
                 err.rate.rf[4,],
                 err.rate.rf[5,],
                 err.rate.rf[6,],
                 err.rate.rf[7,],
                 err.rate.rf[8,],
                 err.rate.rf[9,],
                 err.rate.rf[10,],
                 err.rate.rf[11,])
rownames(rf.err) <- c("A","B","C","D","E","F","G","H","I","W","X","Y")                 
err.rate.rf <- rf.err






