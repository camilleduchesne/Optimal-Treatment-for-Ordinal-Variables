# Camille Duchesne
# March 17th 2022

#Libraries used 

library(caret) #used to compute accuracy from confusion matrix
library(VGAM)
library(MASS)
library(randomForest) 
library(rfUtilities)
library(e1071)
library(caTools)
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(elmNNRcpp)
library(plyr)
library(dplyr)
library(cvms)
library(ggplot2)

#------------------------------Data Preping & Cleaning-------------------------------------------------
data <- read.table(file = 'datrain.txt', header = TRUE)
#head(data)
#data <- na.omit(data), there aren't any missing variables
test <- read.table(file = 'dateststudent.txt', header = TRUE) #this dataset is only necessary for the class competition 
#head(test)

#Splitting the data dataset into Training & Validation sets using a 70/30 split
trainRowCount <- floor(0.7 * nrow(data))
set.seed(1)
trainIndex <- sample(c(1:nrow(data)),size=trainRowCount, replace=FALSE)
train <- data[trainIndex, ]
valid <- data[-trainIndex, ]
train %>% count(y, sort=TRUE)
valid %>% count(y, sort=TRUE)
#Benchmark all values to the highest class
benchmark=(279/600)*100
benchmark

#-----------------------------------------------------------------------------------------------
#------------------------------Naïve Methods----------------------------------------------------
#-----------------------------------------------------------------------------------------------

#---------------------Logistic Regression with multinomial classes------------------------------
#https://machinelearningmastery.com/linear-classification-in-r/
data_LR=train
valid_LR=valid

# CV
set.seed(123) 
train.control <- trainControl(method = "cv", number = 5)

fit <- vglm(y~., family=multinomial, data=data_LR,trControl=train.control)
summary(fit)
probabilities <- predict(fit, data_LR[,1:11], type="response")
predictions <- apply(probabilities, 1, which.max)
confusion_matrix_LR=table(predictions, data_LR$y) 
print(confusion_matrix_LR) #This method seams to have trouble differicienting between classes 1 & 2. 
confusionMatrix(confusion_matrix_LR)
#Accuracy = 0.58 ON TRAINING SET

#VALID ACCURACY
probabilities_valid <- predict(fit, valid_LR[,1:11], type="response")
predictions_valid <- apply(probabilities_valid, 1, which.max)
confusion_matrix_LR=table(predictions_valid, valid_LR$y) 
print(confusion_matrix_LR) #This method seams to have trouble differicienting between classes 1 & 2. 
confusionMatrix(confusion_matrix_LR) #STILL 0.58 Accuracy

#-------------------------Linear Discriminant Analysis------------------------------
data_LDA=train
LDA <- lda(y~., data=data_LDA)
summary(LDA)
predictions_LDA <- predict(LDA, valid[,1:11])$class
cm_LDA =table(predictions_LDA, valid$y) 
print(cm_LDA) 
confusionMatrix(cm_LDA)
#Accuracy = 0.5783


#--------------------------------Random Forest----------------------------------
#trControl <- trainControl(method = "cv",number = 10, search = "grid")
data_RF= train
data_RF$y <- as.factor(data_RF$y) #For the random Forrest, the y needed to be changed into a factor
valid_RF = valid
valid_RF$y <- as.factor(valid_RF$y)

set.seed(177445)
RF_1 <- randomForest(y~., data=data_RF, importance=TRUE) 
RF_2 <- randomForest(y~., data = data_RF, ntree = 500, mtry = 2, importance = TRUE)

#RF_1 is the winning model with the highest right classification rate!
# Predicting on Validation set
predRF_1_VALID <- predict(RF_1, valid[-12], type = "class")
mean(predRF_1_VALID == valid$y)  #Classification rate of 0.66                
cm_RF= table(predRF_1_VALID,valid$y)
print(cm_RF) 

#--------------------------------Visualisations----------------------------------
#to produce the confusion matrix table
#I borrowed this code https://cran.r-project.org/web/packages/cvms/vignettes/Creating_a_confusion_matrix.html
d_multi <- tibble("target" = valid$y,
                  "prediction" = predRF_1_VALID)
d_multi

conf_mat <- confusion_matrix(targets = d_multi$target,
                             predictions = d_multi$prediction)

conf_mat
plot_confusion_matrix(conf_mat$`Confusion Matrix`[[1]],palette = "Oranges")

predRF_1_TEST <- data.frame(as.numeric(predict(RF_1, test, type = "class")))
predRF_1_TEST <- predict(RF_1, test)
write.table(predRF_1_TEST,file = "Camille_Duchesne_2.txt",col.names = F, row.names = F, fileEncoding = "UTF-8")
#write.table(predRF_1_TEST,file = "Camille_Duchesne_2.csv",row.names = F, col.names =F , fileEncoding = "UTF-8")

#write.table(predRF_1_TEST,file = "/Users/camilleduchesne/Documents/1. MSc /H2022/Advanced Stats Learning/assignment_H2022/Camille_Duchesne_2.txt",row.names = FALSE,col.names=FALSE, fileEncoding = "UTF-8")


#Looking at the performance of the 2nd random Forrest
predRF_1_VALID <- predict(RF_2, valid, type = "class")
mean(predRF_1_VALID == valid$y)  #Classification rate of 0.6616667  (slightly worst of than the previous one)                 
table(predRF_1_VALID,valid$y) #65.667x#these values change depending on the sed, 65.66 was the value I obtained 
#at the time i wrote the report

#CV 1 TREE (5 fold)
numFolds <- trainControl(method = "cv", number = 5)
cpGrid <- expand.grid(.cp = seq(0.01, 0.5, 0.01))
train(y ~ ., data = data_RF, method = "rpart", trControl = numFolds, tuneGrid = cpGrid)

#Creating a new CART model using the picked parameter
TreeCV <- rpart(y ~ ., data = data_RF, method = "class", cp = 0.07)
predictionCV <- predict(TreeCV, newdata = valid_RF, type = "class")
table(predictionCV,valid_RF$y)
mean(predictionCV == valid$y) #Ouf 0.5416667
prp(TreeCV)
# 54.1667

#-------------------------------------SVM----------------------------------------
#From the article, I expect the SVP to be the best performing of the Naive methods
#https://odsc.medium.com/build-a-multi-class-support-vector-machine-in-r-abcdd4b7dab6
#Performance of SVM without Cross-Validation
SVM1 = svm(y~., data = data_RF, method="C-classification", kernal="radial", gamma=0.1, cost=10)
SVM1_pred <- predict(SVM1, newdata = valid_RF)
cm_SVM1 = table(valid_RF$y, SVM1_pred)
print(cm_SVM1) 
confusionMatrix(cm_SVM1) # in accuracy 0.585 

#Performance of SVM with Cross-Validation
#https://rpubs.com/markloessi/506713 I used their loop for the cross validation!
folds = createFolds(data_RF$y, k = 5)
TEST_y = lapply(folds, function(x) { 
  training_fold = data_RF[-x, ] 
  test_fold = data_RF[x, ] 
  classifier = svm(formula = y ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')

  #y_pred = predict(classifier, newdata = valid_RF[-12])
  y_pred_TEST = predict(classifier, newdata = test)
  #cm = table(valid_RF[, 12], y_pred)
  #accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  #return(accuracy)
  return(y_pred_TEST)
})
TEST_y
df<- data.frame
Fold_1<-data.frame(as.numeric(TEST_y$Fold1))
Fold_2<-data.frame(as.numeric(TEST_y$Fold2))
Fold_3<-data.frame(as.numeric(TEST_y$Fold3))
Fold_4<-data.frame(as.numeric(TEST_y$Fold4))
Fold_5<-data.frame(as.numeric(TEST_y$Fold5))

df$sum_folds=Fold_1+Fold_2+Fold_3+Fold_4+Fold_5
df$pred=as.factor(ifelse(df$sum_folds<=7,1,(ifelse(df$sum_folds>=13,3,2))))
df <- df["pred"]
#the general principle is that with the predictions from the folds if the sum is smaller than 7, then it accounts
#for all possibilities where 1 is in majority and similarly for 3 
#this implies that the model would only make +/-1 errors in classification

#Let us explore further the errors our model makes
folds = createFolds(data_RF$y, k = 5)
errors = lapply(folds, function(x) { 
  training_fold = data_RF[-x, ] 
  test_fold = data_RF[x, ] 
  classifier = svm(formula = y ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  
  y_pred = predict(classifier, newdata = valid_RF[-12])
  y_pred_TEST = predict(classifier, newdata = test)
  cm = table(valid_RF[, 12], y_pred)
  return(cm)
})
errors
confusionMatrix(errors$Fold3)#  Average accuracy is around 60% not much better than before
#improvement of 2 points using 5fold cross-validation for SVM,
#but the accuracy is still quite low at 60%

#-----------------------------------------------------------------------------------------------
#--------------------------Nominal Binary decomposition approaches------------------------------

#---------------------------CART (Tree, using one-vs-All binary decomposition)------------------
#Isolating 1rst class
data_2 = train
data_2$y <- ifelse(data_2$y==1, 1, 0)
#data_2$y <- as.factor(data_2$y)

#Isolating 2nd class
data_2_2 = train
data_2_2$y <- ifelse(data_2_2$y==2, 1, 0)
#data_2_2$y <- as.factor(data_2_2$y)

#Isolating 3rd class
data_2_3 = train
data_2_3$y <- ifelse(data_2_3$y==3, 1, 0)
#data_2_3$y <- as.factor(data_2_3$y)

CART_1 <- rpart(y~., data_2, method="class")
CART_2 <- rpart(y~., data_2_2, method="class")
CART_3 <- rpart(y~., data_2_3, method="class")

#Get the probability of each class
prob_1 <- predict(CART_1, valid[,1:11], type = "prob")[,2]
prob_2  <- predict(CART_2, valid[,1:11], type = "prob")[,2]
prob_3  <- predict(CART_3, valid[,1:11], type = "prob")[,2]
#Group the probabilities together
prob_MASTER <- data.frame(prob_1,prob_2,prob_3) 

#Find the maximum probability and return the right class
pred_CART <- as.factor(t(apply(prob_MASTER,1,which.max)))
mean(pred_CART == valid$y)             
table(pred_CART,valid$y)
#Awefull accuracy of 0.55 #impossible to classify the 2nd class!! 

#---------------------------SVMs (using OVA binary decomposition)-----------------------------
#Using the same data as prior section (data_2, data_2_2 & data_2_3) isolating each variable
#I'm currently looking for a model who can identify the 3nd class better.... 
#This section will take a little time to run +/- 3 minutes
SVM_1_bd <- svm(y~., data = data_2, kernel = "linear", cost = 10, scale = FALSE, probability=TRUE)
SVM_2_bd <- svm(y~., data = data_2_2, kernel = "linear", cost = 10, scale = FALSE, probability=TRUE)
SVM_3_bd <- svm(y~., data = data_2_3, kernel = "linear", cost = 10, scale = FALSE, probability=TRUE)


#https://odsc.medium.com/build-a-multi-class-support-vector-machine-in-r-abcdd4b7dab6
SVM1 = svm(y~., data = data_RF, method="C-classification", kernal="radial", gamma=0.1, cost=10)
SVM1_pred <- predict(SVM1, newdata = valid_RF)
cm_SVM1 = table(valid_RF$y, SVM1_pred)
print(cm_SVM1) 
confusionMatrix(cm_SVM1) # in accuracy 0.585 

#Get the probability of each class
prob_1 <- predict(SVM_1_bd, valid[,1:11], probability=TRUE)
prob_2  <- predict(SVM_2_bd, valid[,1:11], probability=TRUE)
prob_3  <- predict(SVM_3_bd, valid[,1:11], probability=TRUE)
#Group the probabilities together
prob_MASTER <- data.frame(prob_1,prob_2,prob_3) 

#Find the maximum probability and return the right class
pred_SVM_ovr <- as.factor(t(apply(prob_MASTER,1,which.max)))
mean(pred_SVM_ovr == valid$y)             
table(pred_SVM_ovr,valid$y)
#accuracy of 0.565 

#Seems like this method (ordinal binary decomposition) seems to always have 1 class it has more difficulty pin-pointing

#--------------------------------OnevsOne Binary decomposition Method-----------------------
#I will hand-code the different datasets myself, we want m(m-1)/2 where m is our number of classes (in whis case 3)
data_OVO= train
data_OVO$y <- as.factor(data_OVO$y) #For the random Forrest, the y needed to be changed into a factor
valid_OVO = valid
valid_OVO$y <- as.factor(valid_OVO$y)

#Removed 1 class from each subsets
OVO_1 <- data_OVO[data_OVO$y!= 1,] #Removed class 1 (2 vs 3)
OVO_2 <- data_OVO[data_OVO$y!= 2,] #Removed class 2 (1 vs 3)
OVO_3 <- data_OVO[data_OVO$y!= 3,] #Removed class 3 (1 vs 2)

#Transorfing into binary classes (0 or 1s)
#OVO_1$y <- ifelse(OVO_1$y==2, 1, 0) #CLASS 1:2, 0:3
#OVO_2$y <- ifelse(OVO_2$y==3, 1, 0) #CLASS 1:3, 0:1
#OVO_3$y <- ifelse(OVO_3$y==1, 1, 0) #CLASS 1:1, 0:2

#Training SVM on OVO binary decomposition structure
SVM_1_ovo<- svm(y~., data = OVO_1, kernel = "linear", cost = 100, scale = FALSE, probability=TRUE)
SVM_2_ovo <- svm(y~., data = OVO_2, kernel = "linear", cost = 100, scale = FALSE, probability=TRUE)
SVM_3_ovo <- svm(y~., data = OVO_3, kernel = "linear", cost = 100, scale = FALSE, probability=TRUE)

prob_1_ovo <- predict(SVM_1_ovo, valid[,1:11], probability=TRUE)
prob_2_ovo  <- predict(SVM_2_ovo, valid[,1:11], probability=TRUE)
prob_3_ovo  <- predict(SVM_3_ovo, valid[,1:11], probability=TRUE)

prob_MASTER_ovo <- data.frame(prob_1_ovo,prob_2_ovo,prob_3_ovo) 
prob_MASTER_ovo_2 <- data.frame(prob_1_ovo,prob_2_ovo,prob_3_ovo,valid$y) 

pred_SVM_OVO <- as.factor(t(apply(prob_MASTER_ovo,1,which.max)))
mean(pred_SVM_OVO == valid$y)      # 0.505        
table(pred_SVM_OVO,valid$y)

#When I look at the data outputs from prob_MASTER_ovo, and compare with the actually values for the
#the y of the validation set, I can see that the prob_2_ovo can often correctly identify the 3
#which seems to be the issue in this precious test. I will give this more power to this classifier
pred_SVM_OVO_2= data.frame(valid[,12]) #ATTENTION' they dont have the same indexes
pred_SVM_OVO_2$y_pred =0
prob_MASTER_ovo_F$y_pred=0
#pred_SVM_OVO_2$y_pred<- if(prob_MASTER_ovo$prob_1_ovo==prob_MASTER_ovo$prob_2_ovo || prob_MASTER_ovo$prob_1_ovo==prob_MASTER_ovo$prob_3_ovo ||prob_MASTER_ovo$prob_2_ovo==prob_MASTER_ovo$prob_3_ovo)
prob_MASTER_ovo_F =prob_MASTER_ovo_2
prob_MASTER_ovo_F$prob_1_ovo <-as.factor(prob_MASTER_ovo$prob_1_ovo)
prob_MASTER_ovo_F$prob_2_ovo <-as.factor(prob_MASTER_ovo$prob_2_ovo)
prob_MASTER_ovo_F$prob_3_ovo <-as.factor(prob_MASTER_ovo$prob_3_ovo)

#Loop to return the majority class

for(i in 1:nrow(valid)){
  if(prob_MASTER_ovo_F$prob_2_ovo[i]==prob_MASTER_ovo_F$prob_3_ovo[i]){
    prob_MASTER_ovo_F$y_pred[i]=prob_MASTER_ovo_F$prob_2_ovo[i]
  } else if(prob_MASTER_ovo_F$prob_1_ovo[i]==prob_MASTER_ovo_F$prob_3_ovo[i]){
    prob_MASTER_ovo_F$y_pred[i]=prob_MASTER_ovo_F$prob_1_ovo[i]
  } else if(prob_MASTER_ovo_F$prob_1_ovo[i]==prob_MASTER_ovo_F$prob_2_ovo[i]){
    prob_MASTER_ovo_F$y_pred[i]=prob_MASTER_ovo_F$prob_1_ovo[i]
  } else {
    NULL
  }
}

prob_MASTER_ovo_F$acc <-ifelse(prob_MASTER_ovo_F$valid.y==prob_MASTER_ovo_F$y_pred,1,0)
mean(prob_MASTER_ovo_F$acc) #0.5833333 or #0.54166
table(prob_MASTER_ovo_F$y_pred,valid$y) #0.5533


#-----------------------------------------------------------------------------------------------
#--------------------------Ordinal Binary decomposition approaches------------------------------
#-----------------------------------------------------------------------------------------------

#-----------------Ordinal Binary decompositions----OrderedPartitions----------------------------
data_OP= train
data_OP$y <- as.factor(data_OP$y) #For the random Forrest, the y needed to be changed into a factor
valid_OP = valid
valid_OP$y <- as.factor(valid_OP$y)

#Ordinal PARTITION 
#Follows this matrix accoring to the paper
#     1  2
#  1  0  0
#  2  1  0
#  3  1  1

OP_1 = data_OP
OP_1$y <- ifelse(OP_1$y==2, 1, (ifelse(OP_1$y==3,1,0)))

OP_2 = data_OP
OP_2$y <- ifelse(OP_2$y==3, 1, 0)

#SVMOP
SVM_1_OD <- svm(y~., data = OP_1, kernel = "linear", cost = 10, scale = FALSE)
SVM_2_OD <- svm(y~., data = OP_2, kernel = "linear", cost = 10, scale = FALSE)

#Get the probability of each class
P1_OD <- predict(SVM_1_OD, valid[,1:11])
P2_OD  <- predict(SVM_2_OD, valid[,1:11]) 

#Group the probabilities together
P_MASTER_OD <- data.frame(P1_OD,P2_OD)

Prob_C1_OP <- 1-P_MASTER_OD$P1_OD                #Probability of class 1
Prob_C2_OP <-P_MASTER_OD$P1_OD-P_MASTER_OD$P2_OD #Probability of class 2
Prob_C3_OP <-P_MASTER_OD$P2_OD                   #Probability of class 3

#Get the corresponding probability as per each class
pred_SVM_OP <- as.factor(t(apply(data.frame(Prob_C1_OP,Prob_C2_OP,Prob_C3_OP),1,which.max)))
mean(pred_SVM_OP == valid$y)             
table(pred_SVM_ovr,valid$y)
#Prediction accuracy 0.5016667

#-----------------Extreme Learning Machines----OrderedPartitions----------------------------
ELM_Op <-elm(y~., data = data_OP, nhid = 20, actfun="sig")
pred_classes <- predict(ELM_Op, newdata = valid[,1:11])
pred_probabilities <- predict(ELM_Op, newdata = valid[,1:11], type="prob")
mean(pred_classes == valid$y)             
table(pred_classes,valid$y) #50.166% right classification 

#Change the name of the columns (currently 1, 2, 3)
pred_probabilities = data.frame(pred_probabilities)
pred_probabilities$y_pred_Elm =0
#Try to take a lower threshold for the thrid class! 
pred_probabilities$y_pred_Elm <- ifelse(pred_probabilities$X3>0.44,3,(ifelse(pred_probabilities$X1>pred_probabilities$X2,1,2)))
count(pred_probabilities$y_pred_Elm)
mean(pred_probabilities$y_pred_Elm == valid$y)             
table(pred_probabilities$y_pred_Elm,valid$y) 
#This was the best accuracy I could do with a threshold of (0.44) 51.1667%

#count how many 3 there actually are out of the 600 sample in the validation test 

#-----------------------------------------------------------------------------------------------
#--------------------------------------Threshold Models-----------------------------------------
#-----------------------------------------------------------------------------------------------

#--------------------------------Proportionnal Odds Model (POM)---------------------------------
POM <- polr(y~., data=data_OP, method='loglog')
Pred_POM <-predict(POM,valid_OP[,1:11], type="class")
mean(Pred_POM == valid_OP$y)             
table(Pred_POM,valid_OP$y)
length(Pred_POM)
#Accuracy of 0.58


#-----------------------------------------------------------------------------------------------
#-------------------------Testing Random Forests on binary decomposition------------------------
#-----------------------------------------------------------------------------------------------
#OVA
#Recall these variables were called data_2, data_2_2 & data_2_3
RF_OVA_1 <- randomForest(y~., data=data_2, importance=TRUE) 
RF_OVA_2 <- randomForest(y~., data=data_2_2, importance=TRUE) 
RF_OVA_3 <- randomForest(y~., data=data_2_3, importance=TRUE) 
#RF_2 <- randomForest(y~., data = data_RF, ntree = 500, mtry = 2, importance = TRUE)

# Predicting on Validation set
pred_RF_1_OVA <- predict(RF_OVA_1, valid, type = "class")
pred_RF_2_OVA <- predict(RF_OVA_2, valid, type = "class")
pred_RF_3_OVA<- predict(RF_OVA_3, valid, type = "class")
pred_RF_OVA <- as.factor(t(apply(data.frame(pred_RF_1_OVA,pred_RF_2_OVA,pred_RF_3_OVA),1,which.max)))

mean(pred_RF_OVA == valid$y)  #Classification rate of 0.661667
table(pred_RF_OVA,valid$y)
#(only 0.001667 better than the previous random forrest )







