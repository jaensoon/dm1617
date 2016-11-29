##
# Data Mining Labor 3 Aufgabe 4
#
library(caret)
library(randomForest)
library(e1071)
library(ROCR)

install.packages('caret')
install.packages('randomForest')
install.packages('e1071')
install.packages('ROCR')


# Load Data ---------------------------------------------------------------
# Data source is 
# https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)
# Some preprocessing of the original data set was done -> ../data/vehicle.csv
veh <- read.csv("/home/jwendling/Seafile/Study/Master/Semester 2/Data Mining/Lab/dm1617/data/vehicle.csv", header = TRUE, sep = " ", quote = "\"")
veh

# ------------------------------------------------------------------- Create Training/Test Data

# Spliting data as training and test set. 
# Using createDataPartition() function from caret
set.seed(1711)
inTraining <- createDataPartition(veh$Class, p = .75, list = FALSE)
veh_train <- veh[inTraining,]
veh_test  <- veh[-inTraining,]
table(veh_train$Class)
table(veh_test$Class)

round(prop.table(table(veh_train$Class)),2)
round(prop.table(table(veh_test$Class)),2)

# ------------------------------------------------------------------- Model SVM

set.seed(1711)
tune.out =  tune(svm, Class ~ ., data = veh_train, 
                 kernel = "polynomial", 
                 ranges = list(cost = 10^seq(-2, 1, by = 0.25)))

summary(tune.out)

svm_veh = svm(Class ~ ., kernel = "polynomial", data = veh_train, 
               cost = tune.out$best.parameters$cost)

train.pred = predict(svm_veh, veh_train)
table(veh_train$Class, train.pred)

test.pred = predict(svm_veh, veh_test)
table(veh_test$Class, test.pred)


# ------------------------------------------------------------------- RandomForest

rf_vehicle <- randomForest(formula=Class~., data=veh, ntree=500, mtry=4, do.trace=100)
rf_vehicle

# ------------------------------------------------------------------- kNN

set.seed(1711)
ctrl <- trainControl(method = "repeatedcv", repeats = 1) 

knn_train_z <- as.data.frame(scale(veh_train[-19]))
knn_test_z <- as.data.frame(scale(veh_test[-19]))
knn_train_z$Class <- veh[inTraining, 19]
knn_test_z$Class <- veh[-inTraining, 19]

knn.z2 <- train(Class ~ ., data = knn_train_z, method = "rf", preProcess = "scale", trControl = ctrl, tuneLength = 5)

knn_Predict_z2 <- predict(knn.z2, newdata = veh_test)
confusionMatrix(knn_Predict_z2, veh_test$Class )

# ------------------------------------------------------------------- Compare the Models

test.pred = predict(svm_veh, veh_test)
table(veh_test$Class, test.pred)

print(rf_vehicle)
rf_vehicle$confusion

knn_Predict_z2 <- predict(knn.z2, newdata = veh_test)
confusionMatrix(knn_Predict_z2, veh_test$Class )


# ------------------------------------------------------------------- ROCR

?unname

score <- unname(knn_Predict_z2[, c("Class")])
?prediction
pred <- prediction(knn_Predict_z2, veh_test)
nbperf <- performance(pred, "tpr", "fpr")
plot(nbperf)

pred <- prediction(ROCR.simple$predictions,ROCR.simple$labels)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf)