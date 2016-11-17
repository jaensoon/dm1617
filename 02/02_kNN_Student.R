# Lab: k-NN

# Libraries -------------------------------------------------------------------- 
# Caret is a great(!) R package which provides general interface to nearly
# 150 ML algorithms. It also provides great functions to sample the data 
# (for training and testing), preprocessing, evaluating the model etc.
# To get familiar with caret package, please check following URLs
# http://cran.r-project.org/web/packages/caret/vignettes/caret.pdf
# Also check the caret.r in the tutorial repo
library(caret)
library(randomForest)
library(e1071)
install.packages('caret')
install.packages('randomForest')
install.packages('e1071')

# Functions --------------------------------------------------------------------
normalize <- function(x) {
  return ((x-min(x)) / (max(x)-min(x)))
}

# Load Data ---------------------------------------------------------------
# Data source is 
# https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)
# Some preprocessing of the original data set was done -> ../data/vehicle.csv
veh <- read.csv("/home/jwendling/Seafile/Study/Master/Semester 2/Data Mining/Lab/dm1617/data/vehicle.csv", header = TRUE, sep = " ", quote = "\"")
str(veh)

# Spliting data as training and test set. 
# Using createDataPartition() function from caret
set.seed(1711)
inTraining <- createDataPartition(veh$Class, p = .75, list = FALSE)
df_train <- veh[inTraining,]
df_test  <- veh[-inTraining,]
table(df_train$Class)
table(df_test$Class)

round(prop.table(table(df_train$Class)),2)
round(prop.table(table(df_test$Class)),2)

# Preprocessing  ---------------------------------------------------------------
# Why is it important to normalize numeric data for k-NN?
# Your answer ...Das der Wertebereich sich zwischen 0 und 1 befindet, um zwei
# Skaalen eine Anpassung zu treffen.


# Models -----------------------------------------------------------------------
set.seed(1711)
ctrl <- trainControl(method = "repeatedcv", repeats = 1) 

# Train a model without normalization
?train
knn <- train(Class ~ ., data = df_train, method = "rf", trControl = ctrl, tuneLength = 5)
knn
plot(knn)

# Train a model with normalization 
# this applies the function normalize to columns 1-18
df_train_n <- as.data.frame(lapply(df_train[1:18], normalize))
df_test_n <- as.data.frame(lapply(df_test[1:18], normalize))

# We have excluded the class label during normalization
# For training the k-NN model, we will need to add these class labels, 
# split between the training and test datasets.
df_train_n$Class <- veh[inTraining, 19]
df_test_n$Class <- veh[-inTraining, 19]

knn.n <- train(Class ~ ., data = df_train_n, method = "rf", trControl = ctrl, tuneLength = 5)

# Output of kNN.n
knn.n
plot(knn.n)

# Train a model with standardization (z-Transformation)
# We use the scale function
df_train_z <- as.data.frame(scale(df_train[-19]))
df_test_z <- as.data.frame(scale(df_test[-19]))
df_train_z$Class <- veh[inTraining, 19]
df_test_z$Class <- veh[-inTraining, 19]

knn.z1 <- train(Class ~ ., data = df_train_z, method = "rf", trControl = ctrl, tuneLength = 5)

# Output of kNN.z1
knn.z1
plot(knn.z1)

# Train has also a parameter preprocess. 
# What does this mean?
# Your Answer: Vorbereitung der Daten
# ...
# How could you use preprocess for z-transformation
# Your Answer: Kann verwendet werden, um die Eingabewerte zu standardisieren, damit die Varianz zw. 0 und 1 liegt.
# ...

# Wenn man den preProcess vorher machen möchte
#def_train_z2 <- preProcess(df_train, method = c("scale"))

knn.z2 <- train(Class ~ ., data = df_train_z, method = "rf", preProcess = "scale", trControl = ctrl, tuneLength = 5)

# Plotting yields Number of Neighbours Vs accuracy (based on repeated cross validation)
knn.z2
plot(knn.z2)

# Compare the Models -----------------------------------------------------------

knn_Predict <- predict(knn, newdata = df_test)
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knn_Predict, df_test$Class )

knn_Predict_n <- predict(knn.n, newdata = df_test_n)
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knn_Predict_n, df_test_n$Class )

knn_Predict_z1 <- predict(knn.z1, newdata = df_test_z)
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knn_Predict_z1, df_test_z$Class )

knn_Predict_z2 <- predict(knn.z2, newdata = df_test)
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knn_Predict_z2, df_test$Class )

# Compare the best k-NN Model with your Random Forest solution of Assignment 1-----------------
rf_vehicle <- randomForest(formula=Class~., data=veh, ntree=500, mtry=4, do.trace=100)
print(rf_vehicle)
rf_vehicle$confusion

# RandomForest ist für diese Auswertung besser geeignet