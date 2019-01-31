############# Reading Data #############
### Regression data
diff1 <- read.table('n3train.txt')
diff <- as.data.frame(t(diff1))
colnames(diff) <- c('diff')

feature <- read.table('u10train.txt')
feature <- as.data.frame(t(feature))

seareg <- as.data.frame(cbind(feature,diff))
rownames(seareg) <- c(1:480)

set.seed(1)
train <- sample(480,320,replace=F)
seareg.train <- seareg[train,]
seareg.test <- seareg[-train,]

### Classification data
diff.cat1 <- read.table('n3train_cat.txt')
diff.cat <- as.factor(t(diff.cat1))

seaclass <- as.data.frame(cbind(feature,diff.cat))
rownames(seaclass) <- c(1:480)

set.seed(1)
train <- sample(480,320,replace=F)
seaclass.train <- seaclass[train,]
seaclass.test <- seaclass[-train,]

### Check data
x <- 1:480
y <- as.numeric(t(diff))
par(mfrow=c(1,1))
plot(x,y,type='l')

### Choosing Principle Components ###
pca.train <- prcomp(seaclass.train[,-1594], scale. = T,center=T)
sd.pca <- pca.train$sdev
var.pca <- sd.pca^2
prop.pca <- var.pca/sum(var.pca)
par(mfrow=c(1,2))
plot(prop.pca)
plot(cumsum(prop.pca)) ### Use 150 components from the plot

### Use to regression
train.pca0 <- data.frame(diff= seareg.train$diff, pca.train$x)
train.pca0 <- train.pca0[,1:151] ### First 150 components 
test.pca0 <- predict(pca.train,newdata=seareg.test[,-1594])
test.pca0 <- as.data.frame(test.pca0)
test.pca0 <- test.pca0[,1:150]
test.pca0 <- data.frame(cbind(diff=seareg.test$diff,test.pca0))

### Use to classification
train.pca <- data.frame(diff.cat = seaclass.train$diff.cat, pca.train$x)
train.pca <- train.pca[,1:151] ### First 150 components 
test.pca <- predict(pca.train,newdata=seaclass.test[,-1594])
test.pca <- as.data.frame(test.pca)
test.pca <- test.pca[,1:150]
test.pca <- data.frame(cbind(diff.cat=seaclass.test$diff.cat,test.pca))

############### Regression ###################
train.x<- model.matrix(diff~., train.pca0)
train.y<- train.pca0$diff
test.x <- model.matrix(diff~., test.pca0)
test.y <- test.pca0$diff
grid <- 10 ^ seq(10, -2, length=100)

### Rigid ###
library(glmnet)
set.seed(1)
cv.ridge <- cv.glmnet(train.x, train.y, lambda=grid, alpha=0, nfolds = 5)
bestlam.ridge <- cv.ridge$lambda.min
ridge.pred <- predict(cv.ridge, newx=test.x , s=bestlam.ridge)
error.ridge <- sum((test.y-ridge.pred)^2)
error.ridge
### [1] 101.2943

### Lasso ###
set.seed(1)
cv.lasso <- cv.glmnet(train.x, train.y, lambda=grid, alpha=1, nfolds = 5)
bestlam.lasso <- cv.lasso$lambda.min
lasso.pred <- predict(cv.lasso, newx=test.x , s=bestlam.lasso)
error.lasso <- sum((test.y-lasso.pred)^2)
error.lasso
### [1] 94.09391

### PCR ###
library(pls)
set.seed(1)
pcr.sea <- pcr(diff~., data=train.pca0, sacle=T, validation='CV')
validationplot(pcr.sea, val.type='MSEP',main='PCR model')
sea.pcr.pred <- predict(pcr.sea, test.pca0, ncomp=25)
error.pcr <-sum((test.pca0$diff-sea.pcr.pred)^2)
error.pcr
### [1] 86.65253

### PLS ###
library(pls)
set.seed(1)
pls.sea <- plsr(diff~., data=train.pca0, sacle=T, validation='CV')
summary(pls.sea)
validationplot(pls.sea, val.type='MSEP',main='PLS model')
sea.pls.pred <- predict(pls.sea, test.pca0, ncomp=4)
error.pls <-sum((test.pca0$diff-sea.pls.pred)^2)
error.pls
### [1] 78.2548

### CART with Bagging ###
library(randomForest)
set.seed(1)
bag.sea <- randomForest(diff~.,data=train.pca0,mtry=150,importance=T)
sea.bag.pred <- predict(bag.sea, test.pca0) 
error.sea.bag <- sum((test.pca0$diff - sea.bag.pred)^2)
error.sea.bag
### [1] 115.3521

### Random Forest ###
set.seed(1)
rf.sea <- randomForest(diff~.,data=train.pca0,importance=T)
sea.rf.pred <- predict(rf.sea, test.pca0) 
error.sea.rf <- sum((test.pca0$diff - sea.rf.pred)^2)
error.sea.rf
### [1] 111.2026

### MARS ###
library(earth)
set.seed(1)
mars.sea <- earth(diff~.,data=train.pca0, pmethod = 'cv', nfold=5)
sea.mars.pred <- predict(mars.sea, test.pca0) 
error.sea.mars <- sum((test.pca0$diff - sea.mars.pred)^2)
error.sea.mars
### [1] 108.0426

### Neural Network ###
library(nnet)
x <- test.pca0[,-1]
y <- test.pca0[,1]

sea.nn <- nnet(diff~.,train.pca0,size=5, linout=T, decay=0, MaxNWts=100000)
sea.nn.pred <- predict(sea.nn,x,type='raw')
error.sea.nn <- sum((y - sea.nn.pred)^2)
error.sea.nn
### [1] 104.8648


############### Classification ###################
### CART with Bagging ###
library(randomForest)
set.seed(1)
bag.sea <- randomForest(diff.cat~.,data=train.pca,mtry=150,importance=T)
sea.bag.pred <- predict(bag.sea, test.pca,type='class') 
ctable.bag <- table(sea.bag.pred,seaclass.test$diff.cat)
ctable.bag
###            truth
###   predict -1  0  1
###        -1 15  3  4
###        0  24 68 38
###        1   2  2  4
bag.rate <- 1-sum(diag(ctable.bag))/length(seaclass.test$diff.cat)
bag.rate
### [1] 0.45625

### Random Forest ###
library(randomForest)
set.seed(1)
rf.sea <- randomForest(diff.cat~.,data=train.pca,importance=T)
sea.rf.pred <- predict(rf.sea, test.pca,type='class') 
ctable.rf <- table(sea.rf.pred,seaclass.test$diff.cat)
ctable.rf
###            truth
###   predict -1  0  1
###        -1  8  2  0
###        0  32 71 46
###        1   1  0  0
rf.rate <- 1-sum(diag(ctable.rf))/length(seaclass.test$diff.cat)
rf.rate
### [1] 0.50625

### Neural Networks ###
library(nnet)
clssea.train <- class.ind(train.pca$diff.cat)
clssea.test <- class.ind(test.pca$diff.cat)

set.seed(1)
nnet.sea <- nnet(train.pca[,-1],clssea.train,size=100,softmax=T,MaxNWts=100000)
sea.nnet.pred <- predict(nnet.sea,clssea.test,type='class')
ctable.nnet <- table(sea.nnet.pred,seaclass.test$diff.cat)
ctable.nnet### Some missing value


### Deep Learning ###
library(h2o)
h2o.init(nthreads = -1)
seaclass.train.h2o <- as.h2o(train.pca)
seaclass.test.h2o <- as.h2o(test.pca)

y <- "diff.cat"
x <- setdiff(names(train.pca),y)

seaclass.train.h2o[,y] <- as.factor(seaclass.train.h2o[,y])
seaclass.test.h2o[,y] <- as.factor(seaclass.test.h2o[,y])

set.seed(1)
deep.sea <- h2o.deeplearning(x=x,y=y,training_frame = seaclass.train.h2o,validation_frame = seaclass.test.h2o,
                             activation = 'Rectifier',hidden = c(200,50),epochs=10,nfolds=5)

h2o.performance(deep.sea,train = T)
h2o.performance(deep.sea,valid = T)

###Confusion Matrix: vertical: actual; across: predicted
###        -1  0  1  Error       Rate
### -1     20 15  6 0.5122 =  21 / 41
### 0       9 56  8 0.2329 =  17 / 73
### 1       5 23 18 0.6087 =  28 / 46
### Totals 34 94 32 0.4125 = 66 / 160


### SVM ###
library(e1071)
set.seed(1)
sea.svm1 <- tune(svm,diff.cat~.,data=train.pca,kernel='linear',ranges=list(cost=c(0.01,0.05,0.1,0.5,1,5,10)))
sea.svm1$best.parameters
###   cost
###   0.01
pred.sea.svm1 <- predict(sea.svm1$best.model,test.pca)
ctable.svm1 <- table(predict=pred.sea.svm1,truth=seaclass.test$diff.cat)
ctable.svm1
###            truth
###   predict -1  0  1
###        -1 18  8  6
###        0  19 62 25
###        1   4  3 15
svm1.rate <- 1-sum(diag(ctable.svm1))/length(seaclass.test$diff.cat)
svm1.rate
### [1] 0.40625

set.seed(1)
sea.svm2 <- tune(svm,diff.cat~.,data=train.pca,kernel='polynomial',degree=2,ranges=list(cost=c(0.01,0.05,0.1,0.5,1,5,10)))
sea.svm2$best.parameters
###   cost
###   0.01
pred.sea.svm2 <- predict(sea.svm2$best.model,test.pca)
ctable.svm2 <- table(predict=pred.sea.svm2,truth=seaclass.test$diff.cat)
ctable.svm2
###            truth
###   predict -1  0  1
###        -1  0  0  0
###        0  41 73 46
###        1   0  0  0
svm2.rate <- 1-sum(diag(ctable.svm2))/length(seaclass.test$diff.cat)
svm2.rate ### Can't distinguish between 1,0 and 1,-1
### [1] 0.54375

set.seed(1)
sea.svm3 <- tune(svm,diff.cat~.,data=train.pca,kernel='radial',ranges=list(cost=c(0.01,0.05,0.1,0.5,1,5,10),gamma=c(0.001,0.01,0.05,0.1,0.5,1,5,10,100)))
sea.svm3$best.parameters
###   cost gamma
###    5   0.001
pred.sea.svm3 <- predict(sea.svm3$best.model,test.pca)
ctable.svm3 <- table(predict=pred.sea.svm3,truth=seaclass.test$diff.cat)
ctable.svm3
###            truth
###   predict -1  0  1
###        -1 20  9  6
###        0  17 61 24
###        1   4  3 16
svm3.rate <- 1-sum(diag(ctable.svm3))/length(seaclass.test$diff.cat)
svm3.rate
### [1] 0.39375

set.seed(1)
sea.svm4 <- tune(svm,diff.cat~.,data=train.pca,kernel='sigmoid',ranges=list(cost=c(0.01,0.05,0.1,0.5,1,5,10),gamma=c(0.001,0.01,0.05,0.1,0.5,1,5,10,100)))
sea.svm4$best.parameters
###   cost gamma
###    1   0.01
pred.sea.svm4 <- predict(sea.svm4$best.model,test.pca)
ctable.svm4 <- table(predict=pred.sea.svm4,truth=seaclass.test$diff.cat)
ctable.svm4
###            truth
###   predict -1  0  1
###        -1 19  7  4
###        0  19 61 29
###        1   3  5 13
svm4.rate <- 1-sum(diag(ctable.svm4))/length(seaclass.test$diff.cat)
svm4.rate
### [1] 0.41875

############### Test data #####################
############# Reading Data #############
### Regression data
diff12 <- read.table('n3test.txt')
diff2 <- as.data.frame(t(diff12))
colnames(diff2) <- c('diff')

feature2 <- read.table('u10test.txt')
feature2 <- as.data.frame(t(feature2))

test.seareg <- as.data.frame(cbind(feature2,diff2))
rownames(test.seareg) <- c(1:74)

### Classification data
diff.cat12 <- read.table('n3test_cat.txt')
diff.cat2 <- as.factor(t(diff.cat12))

test.seaclass <- as.data.frame(cbind(feature,diff.cat2))
rownames(test.seaclass) <- c(1:74)

### Choosing Principle Components ###
pca1.train <- prcomp(seareg[,-1594], scale. = T,center=T)
### Use to training
### reg
train1.pca0 <- data.frame(diff= seareg$diff, pca1.train$x)
train1.pca0 <- train1.pca0[,1:151] ### First 150 components 

train1.pca <- data.frame(diff.cat = seaclass$diff.cat, pca1.train$x)
train1.pca <- train1.pca[,1:151] ### First 150 components 

### Use to regression
test1.pca0 <- predict(pca1.train,newdata=test.seareg[,-1594])
test1.pca0 <- as.data.frame(test1.pca0)
test1.pca0 <- test1.pca0[,1:150]
test1.pca0 <- data.frame(cbind(diff=test.seareg$diff,test1.pca0))

### Use to classification
test1.pca <- predict(pca1.train,newdata=test.seaclass[,-1594])
test1.pca <- as.data.frame(test1.pca)
test1.pca <- test1.pca[,1:150]
test1.pca <- data.frame(cbind(diff.cat=test.seaclass$diff.cat,test1.pca))

########## Using partial training data ################
### Regression --- PLS ###
library(pls)
set.seed(1)
sea.pls.pred <- predict(pls.sea, test1.pca0, ncomp=4)
error.pls <-sum((test1.pca0$diff-sea.pls.pred)^2)
error.pls
### [1] 39.56959

### Classification --- SVM with radial kernel ###
library(e1071)
sea.svm3.final <- svm(diff.cat~.,data=train.pca,kernel='radial',cost=5,gamma=0.001)
pred.sea.svm3 <- predict(sea.svm3.final,test1.pca)
ctable.svm.final <- table(predict=pred.sea.svm3,truth=test1.pca$diff.cat)
ctable.svm.final
###            truth
###   predict -1  0  1
###        -1  2  2  4
###        0  11 31  8
###        1   1  8  7
rate.test.final <- 1-sum(diag(ctable.svm.final))/length(test1.pca$diff.cat)
rate.test.final
### [1] 0.4594595

########## Using All the training data ################
### Regression --- PLS ###
library(pls)
set.seed(1)
pls.sea <- plsr(diff~., data=train1.pca0, sacle=T, validation='CV')
seareg.train.pls.pred <- predict(pls.sea, train1.pca0, ncomp=4)
error.train <-sum((train1.pca0$diff-seareg.train.pls.pred)^2)
error.train
### [1] 196.5112
seareg.test.pls.pred <- predict(pls.sea, test1.pca0, ncomp=4)
error.test <-sum((test.seareg$diff-seareg.test.pls.pred)^2)
error.test
### [1] 34.65071  ### test dataset

### Classification --- SVM with radial kernel ###
library(e1071)
set.seed(1)
sea.svm.final <- svm(diff.cat~.,data=train1.pca,kernel='radial',cost=5,gamma=0.001)
seacls.test.pred <- predict(sea.svm.final,test1.pca)
ctable.test <- table(predict=seacls.test.pred,truth=test1.pca$diff.cat)
ctable.test
###            truth
###   predict -1  0  1
###        -1  7  5  2
###        0   5 32 12
###        1   2  4  5
rate.test <- 1-sum(diag(ctable.test))/length(test.seaclass$diff.cat)
rate.test
### [1] 0.4054054
