# Lab: SVM Sale Data

# Libraries -------------------------------------------------------------------- 
install.packages('ISLR')
library(ISLR) #contains the dataset
library(caret)
library(e1071)


# Load Data --------------------------------------------------------------------
# 
# Description
# The data contains 1070 purchases where the customer either purchased Citrus Hill or Minute Maid
# Orange Juice. A number of characteristics of the customer and product are recorded.
# 
# A data frame with 1070 observations on the following 18 variables.
# Purchase A factor with levels CH and MM indicating whether the customer purchased Citrus Hill or
# Minute Maid Orange Juice
# WeekofPurchase Week of purchase
# StoreID Store ID
# PriceCH Price charged for CH
# PriceMM Price charged for MM
# DiscCH Discount offered for CH
# DiscMM Discount offered for MM
# SpecialCH Indicator of special on CH
# SpecialMM Indicator of special on MM
# LoyalCH Customer brand loyalty for CH
# SalePriceMM Sale price for MM
# SalePriceCH Sale price for CH
# PriceDiff Sale price of MM less sale price of CH
# Store7 A factor with levels No and Yes indicating whether the sale is at Store 7
# PctDiscMM Percentage discount for MM
# PctDiscCH Percentage discount for CH
# ListPriceDiff List price of MM less list price of CH
# STORE Which of 5 possible stores the sale occured at

str(OJ)
head(OJ)
# Rename column Purchase to Class 
names(OJ)[names(OJ)=="Purchase"] <- "Class"
str(OJ)

# Spliting data as training and test set. 
# Using createDataPartition() function from caret
set.seed(1711)
inTraining <- createDataPartition(OJ$Class, p = .75, list = FALSE)
OJ.train <- OJ[inTraining,]
OJ.test  <- OJ[-inTraining,]
table(OJ.train$Class)
table(OJ.test$Class)

round(prop.table(table(OJ.train$Class)),2)
round(prop.table(table(OJ.test$Class)),2)

# SVM Linear Model -------------------------------------------------------------
ctrl <- trainControl(method = "cv", 
                     number=10, 
                     savePred=TRUE, 
                     classProb=TRUE)

set.seed(1711)
svm.linear = train(Class ~ ., data=OJ.train, 
                   method = "svmLinear",
                   preProc = c("center","scale"),
                   trControl = ctrl)
 
svm.linear = svm(Class ~ ., kernel = "linear", data = OJ.train)
 
summary(svm.linear)

# Support vector classifier creates 437 support vectors out of 803 training points. 


# Evaluation -------------------------------------------------------------------
train.pred = predict(svm.linear, OJ.train)
table(OJ.train$Class, train.pred)

conf <- confusionMatrix(train.pred, OJ.train$Class, dnn = c("Prediction", "Actual"))
conf$table
round(conf$overall,2)
round(conf$byClass,2)

test.pred = predict(svm.linear, OJ.test)
table(OJ.test$Class, test.pred)
?confusionMatrix
conf <- confusionMatrix(test.pred, OJ.test$Class, dnn = c("Prediction", "Actual"))
conf$table
conf$overall
round(conf$overall,2)
round(conf$byClass,2)

# The training error rate is 16.9% and test error rate is about 17.8%.

# Tune Model -------------------------------------------------------------------
# 
set.seed(1711)
tune.out = tune(svm, Class ~ ., data = OJ.train, 
                kernel = "linear", 
                ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
summary(tune.out)

# Tuning shows that optimal cost is 0.3162
# Richtiges Ergebnis: 6   0.17782794 0.1607099 0.04479902

# Train tuned linear model -----------------------------------------------------

svm.linear = svm(Class ~ ., kernel = "linear", data = OJ.train, 
                 cost = tune.out$best.parameters$cost)

train.pred = predict(svm.linear, OJ.train)
table(OJ.train$Class, train.pred)

test.pred = predict(svm.linear, OJ.test)
table(OJ.test$Class, test.pred)

# SVM Radial Model -------------------------------------------------------------
set.seed(1771)
?svm
svm.radial = svm(Class ~ ., kernel = "radial", data = OJ.train)

# Tune Radial Model -------------------------------------------------------------------
set.seed(1711)
tune.out = tune(svm, Class ~ ., data = OJ.train, 
                kernel = "radial", 
                ranges = list(cost = 10^seq(-2, 1, by = 0.25)))

summary(tune.out)

# Train tuned radial model -----------------------------------------------------
svm.radial = svm(Class ~ ., kernel = "radial", data = OJ.train, 
                 cost = tune.out$best.parameters$cost)

train.pred = predict(svm.radial, OJ.train)
table(OJ.train$Class, train.pred)

test.pred = predict(svm.radial, OJ.test)
table(OJ.test$Class, test.pred)

# Tuning slightly decreases training error to 14.6% and slightly increases 
# test error to 16% which is still better than linear kernel.
# 

# SVM Polynomial Model -------------------------------------------------------------
set.seed(1711)
svm.ploy = svm(Class ~ ., kernel = "polynomial", data = OJ.train)

# Summary shows that polynomial kernel produces 452 support vectors, out of which, 
# 232 belong to level CH and remaining 220 belong to level MM. This kernel 
# produces a train error of 17.1% and a test error of 18.1% which are slightly 
# higher than the errors produces by radial kernel but lower than the errors 
# produced by linear kernel.

# Tune Model -------------------------------------------------------------------
set.seed(1711)
tune.out =  tune(svm, Class ~ ., data = OJ.train, 
                 kernel = "polynomial", 
                 ranges = list(cost = 10^seq(-2, 1, by = 0.25)))

summary(tune.out)

# Train tuned ploy model -----------------------------------------------------
svm.poly = svm(Class ~ ., kernel = "polynomial", data = OJ.train, 
                   cost = tune.out$best.parameters$cost)

train.pred = predict(svm.poly, OJ.train)
table(OJ.train$Class, train.pred)

test.pred = predict(svm.poly, OJ.test)
table(OJ.test$Class, test.pred)
  
# Compare the models
# Which poduces the best results?
# Are you satisfied with the misclassification error?

