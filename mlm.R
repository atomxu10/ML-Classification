# enviornment
library(ggplot2)
library(tidyverse)
library(readr)
library(pROC)
library(MASS)
library(rpart)
library(rattle)
library(randomForest)
library(xgboost)
library(e1071)
library(caret)
library(irr)
library(lattice)
library(grid)
library(DMwR)
library(neuralnet)
setwd('/Users/atom/R/ml_assignment2')

column<-read.csv(file="gamma.csv",header=TRUE)
str(column)
column$Class <- as.factor(column$class)
column = column[,-11]
column.lr <-column

# delete NULL
removeRowsAllNa  <- function(x){x[apply(x, 1, function(y) any(!is.na(y))),]}
column <- removeRowsAllNa(column)
str(column)
attach(column)

#————————————————————- Logistic regression ——————————————————————

# split in training and testing sets (trainingset= 0.75, testingset=0.25)
set.seed(100)
ind<-sample(2,nrow(column),replace=TRUE,prob=c(0.75,0.25))
columntrain<-column[ind==1,]
columntest<-column[ind==2,]

# data normalization
columntrain.scale <- columntrain
columntrain.scale[1:10] <- scale(columntrain[1:10])
columntest.scale <- columntest
columntest.scale[1:10] <- scale(columntest[1:10])

# train the model
columntrain.lr<-glm(Class~.,data=columntrain.scale,family=binomial)
summary(columntrain.lr)

# predict on the testing set
columntest.prob <- predict(columntrain.lr,columntest.scale,type="response")

## ROC
roc_lr <- roc(columntest$Class, as.numeric(columntest.prob),
              ci = TRUE)
plot.roc(roc_lr, print.auc = TRUE, auc.polygon = TRUE, print.thres = TRUE,
         main = "ROC curve of Logistic Regression")

## compare the Threshold at 0.5 and 0.485 (optimal threshold) and confusion matrix
columntest.pred1<-ifelse(columntest.prob>0.5,1,0)
columntest.pred2<-ifelse(columntest.prob>0.485,1,0)
table(columntest$Class,columntest.pred1)
table(columntest$Class,columntest.pred2)
table_test_lr <- table(columntest$Class,columntest.pred2)

# testing set Accuracy
accuracy_test_lr <- (table_test_lr[1,1]+table_test_lr[2,2])/sum(table_test_lr)
print(accuracy_test_lr)

#—————————————————————— discriminant analysis ————————————————————-
#—————————————————————— LDA ————————————————————
# training set model
columntrain.lda <- lda(Class ~ Length + Width + Size + Conc + Asym + M3Long + M3Trans + Alpha + Dist , data = columntrain.scale)
columntrain.lda

plot(columntrain.lda)
# predict on the testing set
columntest.pred.lda <- predict(columntrain.lda, columntest.scale)
columntest.pred.lda$class
mean(columntest.pred.lda$class == columntest.scale$Class)

# testing Accuracy

accuracy_test_lda <- (table_test_lda[1,1]+table_test_lda[2,2])/sum(table_test_lda)
print(accuracy_test_lda)

# ROC and AUC
roc_lda <- roc(columntest.scale$Class, as.numeric(columntest.pred.lda$class),
              ci = TRUE)
plot.roc(roc_lda, print.auc = TRUE, auc.polygon = TRUE, print.thres = TRUE,
         main = "ROC curve of LDA")

#—————————————————————— QDA ————————————————————
columntrain.qda <- qda(Class~.,data = columntrain.scale)
columntest.pred.qda <- predict(columntrain.qda, columntest.scale)

#testing accuracy
mean(columntest.pred.qda$class == columntest.scale$Class)
table_test_qda <- table(columntest$Class,columntest.pred.qda$class)

accuracy_test_qda <- (table_test_qda[1,1]+table_test_qda[2,2])/sum(table_test_qda)
print(accuracy_test_qda)

# ROC and AUC
roc_qda <- roc(columntest.scale$Class, as.numeric(columntest.pred.qda$class),
               ci = TRUE)
plot.roc(roc_qda, print.auc = TRUE, auc.polygon = TRUE, print.thres = TRUE,
         main = "ROC curve of QDA")

#—————————————————————— decision trees (CART classification tree) ——————————————————————
# training set model
columntrain.cart<-rpart(Class ~., 
                  data=columntrain, 
                  method="class", 
                  parms=list( split="gini" ),
                  control=rpart.control(cp=0.01)) #split = “gini”
printcp(columntrain.cart)

# plot the tree
fancyRpartPlot( columntrain.cart, 
                main = paste("CART"))

# testing set prediction
columntest.cart <- columntest
columntest.pred.cart <- predict(columntrain.cart, columntest.cart, type="class" )   

# test accuracy
5 <- table(columntest.cart$Class,columntest.pred.cart)
accuracy_test_cart <- (table_test_cart[1,1]+table_test_cart[2,2])/sum(table_test_cart)
print(accuracy_test_cart)

# ROC & AUC
roc_cart <- roc(columntest$Class, as.numeric(columntest.pred.cart),
              ci = TRUE)
plot.roc(roc_cart, print.auc = TRUE, auc.polygon = TRUE, print.thres = TRUE,
         main = "ROC curve of CART")

#—————————————————————— Random forest ——————————————————————
n <- 8
errRate <- c(1)
for (i in 1:n){ 
  m <- randomForest(Class~.,data=columntrain,mtry=i,proximity=TRUE) 
  err<-mean(m$err.rate)
  errRate[i] <- err  }  
print(errRate)

# choose mtry=2 
columntrain.rf<-randomForest(Class~.,data=columntrain,mtry=2)
columntest.pred.rf <- predict(columntrain.rf,columntest)

# plot
plot(columntrain.rf)

# test the accuracy
table_test_rf <- table(columntest$Class,columntest.pred.rf)
accuracy_test_rf <- (table_test_rf[1,1]+table_test_rf[2,2])/sum(table_test_rf)
print(accuracy_test_rf)

# ROC & AUC
roc_rf <- roc(columntest$Class, as.numeric(columntest.pred.rf),
               ci = TRUE)
plot.roc(roc_rf, print.auc = TRUE, auc.polygon = TRUE, print.thres = TRUE,
         main = "ROC curve of random forest")

#—————————————————————— XGBoost ————————————————————————
# transform "Class" to numeric
str(column)
column.xgb <- column
columntrain.xgb <- columntrain
columntest.xgb <- columntest

column.xgb$Class <- as.numeric(column.xgb$Class)
columntrain.xgb$Class <- as.numeric(columntrain.xgb$Class)
columntest.xgb$Class <- as.numeric(columntest.xgb$Class)

# transform data frame to a matrix
column.xgb <-as.matrix(column.xgb)
columntrain.xgb <-as.matrix(columntrain.xgb)
columntest.xgb <-as.matrix(columntest.xgb)

# use cross-validation to get nround
column.xgb.cv <-xgb.cv(data=column.xgb[,1:10],label=column.xgb[,11],nrounds=50,nfold=5,verbose=0)
column.xgb.cv

# looks to me like it only needs about 15 rounds to minimise the test mean square error
# xgboost
xgboost_model<-xgboost(data=columntrain.xgb[,1:10],label=columntrain.xgb[,11],nrounds=15,verbose=0)
columntest.pred.xgb <-predict(xgboost_model,newdata=columntest.xgb[,1:10])
columntest.pred.xgb <- round(columntest.pred.xgb)

# test the accuracy
table_test_xgb <- table(columntest.pred.xgb,columntest.xgb[,11])
accuracy_test_xgb <- (table_test_xgb[1,1]+table_test_xgb[2,2])/sum(table_test_xgb)
print(accuracy_test_xgb)

# ROC and AUC
roc_xgb <- roc(columntest$Class, columntest.pred.xgb,
               ci = TRUE)
plot.roc(roc_xgb, print.auc = TRUE, auc.polygon = TRUE, print.thres = TRUE,
         main = "ROC curve of XGBoost")

#—————————————————————— support vector machines ————————————————————————
# use data standardize
str(columntest.scale)
str(columntrain.scale)

model.svm1<-svm(columntrain.scale[,-11],columntrain.scale[,11],type="C-classification")
test.pred1 <- predict(model.svm1, columntest.scale[,-11]) 
mean(test.pred1==columntest.scale[,11])

model.svm2<-svm(columntrain.scale[,-11],columntrain.scale[,11],type="C-classification",kernel = "polynomial")
test.pred2 <- predict(model.svm2, columntest.scale[,-11]) 
mean(test.pred2==columntest.scale[,11])

model.svm3<-svm(columntrain.scale[,-11],columntrain.scale[,11],type="C-classification",kernel = "linear")
test.pred3 <- predict(model.svm3, columntest.scale[,-11]) 
mean(test.pred3==columntest.scale[,11])


gamma_fine<-seq(10^(-1),10^1,1)
cost_fine<-seq(1,100,10)
tObj<-tune.svm(x=columntrain.scale[,-11],y=columntrain.scale[,11],data=columntrain.scale,type="C-classification", 
               gamma = gamma_fine,cost=cost_fine,scale=FALSE)
summary(tObj)

model.svm.best<-svm(columntrain.scale[,-11],columntrain.scale[,11],type="C-classification",gamma = 0.1,cost =10)
test.pred.svm.best <- predict(model.svm.best, columntest.scale[,-11]) 
mean(test.pred.svm.best==columntest.scale[,11])

 # test the accuracy
table_test_svm <- table(test.pred.svm.best,columntest[,11])
accuracy_test_svm <- (table_test_svm[1,1]+table_test_svm[2,2])/sum(table_test_svm)
print(accuracy_test_svm)

# ROC and AUC
roc_svm <- roc(columntest$Class, as.numeric(test.pred.svm.best),
              ci = TRUE)
plot.roc(roc_svm, print.auc = TRUE, auc.polygon = TRUE, print.thres = TRUE,
         main = "ROC curve of SVM")

#—————————————————————— neural networks ———————————————————————

fomula <- "Class~ Length+ Width + Size +Conc +Conc1 +Asym +M3Long +M3Trans+ Alpha+ Dist"
f <-as.formula(fomula)

model.nn<-neuralnet(f,data = columntrain.scale,hidden=2,err.fct="ce",linear.output=FALSE)
plot(model.nn)
head(model.nn$weights)
head(model.nn$generalized.weights)

columntest.pred.nn <- predict(model.nn, columntest.scale[,-11],type="class")
columntest.pred.nn1 <- c(0,1)[apply(columntest.pred.nn,1,which.max)]
columntest.pred.nn2 <-gsub("0", "g",columntest.pred.nn1)
columntest.pred.nn2 <-gsub("1", "h",columntest.pred.nn1)
# test the accuracy
table_test_nn <- table(columntest.pred.nn1,columntest[,11])
accuracy_test_nn <- (table_test_nn[1,1]+table_test_nn[2,2])/sum(table_test_nn)
print(accuracy_test_nn)

# ROC and AUC
roc_nn <- roc(columntest$Class, as.numeric(columntest.pred.nn1),ci = TRUE)
plot.roc(roc_nn, print.auc = TRUE, auc.polygon = TRUE, print.thres = TRUE,
         main = "ROC curve of neural networks")


