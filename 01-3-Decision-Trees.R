# Lab: Decision Trees with rpart and C5.0

# Importing Data ------------------------------------------------------------
# Data source is 
# https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
# Some preprocessing of the original data set was done -> data/credit.csv

# libraries for decision trees
library(rpart)
library(C50)
# for cross tabulation of predicted versus actual classes
library(gmodels)
install.packages('gmodels')

getwd()
credit <- read.csv("/home/jwendling/Seafile/Study/Master/Semester 2/Data Mining/Praktikum/data-mining-lecture-assignments-201617ws/data/credit.csv")
str(credit)

# First analysis and preprocessing ------------------------------------------
# look at two characteristics of the applicant
table(credit$checking_balance)
table(credit$savings_balance)

# look at two characteristics of the loan
summary(credit$months_loan_duration)
summary(credit$amount)

# look at the class variable
table(credit$Class)

# create a random sample for training and test data
# use set.seed to use the same random number sequence as the tutorial
# split training and test dataset
set.seed(1711)
credit.rand <- credit[order(runif(1000)), ]
sample.indizes <- sample(1:nrow(credit), 0.66 * nrow(credit), replace = FALSE)
credit.train <- credit.rand[sample.indizes,]
credit.test <- credit.rand[-sample.indizes,]

# check the proportion of class variable
prop.table(table(credit.train$Class))
prop.table(table(credit.test$Class))

# compare the credit and credit.rand data frames
summary(credit$amount)
summary(credit.rand$amount)
head(credit$amount)
head(credit.rand$amount)

# rpart model ------------------------------------------
?rpart

# Aufgabe 3.1 Fragen ----------------------------------------------------------

### FRAGE: Welche Eigenschaften des Baums können über control gesteuert werden?
#
# minsplit, minbucket, cp, maxcompete, maxsurrogate, usesurrogate,
# xval, surrogatestyle, maxdepth, ...
#

### FRAGE: Was bedeutet xval?
#
# Anzahl der Cross-Validationen
#

tree1 <- rpart(Class ~ .,
               control=rpart.control(minsplit=5, cp=0.0),
               data=credit.train)

# Decision Tree Dendogram
plot(tree1, 
     uniform = TRUE, 
     compress = TRUE, 
     margin = 0.2, 
     branch = 0.3)
# Label on Decision Tree
text(tree1, 
     use.n = TRUE, 
     digits = 3, 
     cex = 0.6)

# Train with XValidation
tree2 <- rpart(Class ~ .,
               control=rpart.control(minsplit = 5, cp = 0.0),
               data=credit.test)


# rpart Evaluating model performance ------------------------------------------
# create a factor vector of predictions on test data
model <- tree2
credit.pred <- predict(tree2, credit.test, type = c("class"))

# cross tabulation of predicted versus actual classes
CrossTable(credit.test$Class, credit.pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Class', 'predicted Class'))

### FRAGE: Wie interpretieren Sie das Ergebnis der Konfisionsmatrix?
#
# Bei insgesamt 340 wurden
#     - 240 als "Nein" vorhergesagt, von diesen waren 234 wirklich "Nein" und 6 "Ja"
#     - 100 als "Ja" vorhergesagt, von diesen waren 90 wirklich "Ja" und 10 "Nein"
#

### FRAGE: Sind Sie mit dem Ergebnis zufrieden?
#
# Ja, größtenteils stimmen die Vorhersagen. Es gibt nur kleine
# Ausreiser. Z. B. bei Nein wurden 234 Vorhersagen richtig gemacht
# und 10 sind falsch.se
#

# Aufgabe 3.2 Fragen ----------------------------------------------------------

# C5.0 Model ------------------------------------------
# build the simplest decision tree
??C5.0

model <- C5.0(x=credit.train[-17], y=credit.train$Class, trials=1, control=C5.0Control())

# display simple facts about the tree
model

# display detailed information about the tree
summary(model)

# Exkurs C5 Importance Measures
C5imp(model, metric = "usage", pct = TRUE)

# C5.0 Evaluating model performance ------------------------------------------
# create a factor vector of predictions on test data
credit.pred <- predict(model, credit.test, type = c("class"))

# cross tabulation of predicted versus actual classes
CrossTable(credit.test$Class, credit.pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Class', 'predicted Class'))

# C5.0 Improving model performance ------------------------------------------
# Boosting the accuracy of decision trees
# boosted decision tree with 10 trials
# boost factor is a hyperparameter. 
# Not a parameter of the model itself but it determines the computing of the modell
credit.boost10 <- C5.0(x=credit.train[-17], y=credit.train$Class, trials=10, control=C5.0Control())
summary(credit.boost10)

credit.boost_pred10 <- predict(credit.boost10, credit.test, type = c("class"))
CrossTable(credit.test$Class, credit.boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Class', 'predicted Class'))

# boosted decision tree with 100 trials 
credit.boost100 <- C5.0(x=credit.train[-17], y=credit.train$Class, trials=100, control=C5.0Control())
summary(credit.boost10)

credit.boost_pred100 <- predict(credit.boost100, credit.test, type = c("class"))
CrossTable(credit.test$Class, credit.boost_pred100,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Class', 'predicted Class'))

# C5.0 Making some mistakes more costly than others ------------------------------------
# create a cost matrix
matrix_dimensions <- list(c("no", "yes"), c("no", "yes"))
names(matrix_dimensions) <- c("predicted", "actual")
error_cost <- matrix(c(0, 2, 5, 0), nrow = 2, dimnames = matrix_dimensions)
error_cost

# apply the cost matrix to the tree
credit.cost <- C5.0(x=credit.train[-17], y=credit.train$Class, trials=10, control=C5.0Control(), costs = error_cost)
credit.cost_predict <- predict(credit.cost, credit.test, type = c("class"))

# Cross table...
CrossTable(credit.test$Class, credit.cost_predict,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Class', 'predicted Class'))

# feel free to try some other parameters
prunetree <- tree1
printcp(prunetree)
plotcp(prunetree)
prunetree2 <- prune.rpart(prunetree, cp=0.0366667)
printcp(prunetree2)
plotcp(prunetree2)

# Testing a fancy tree plotting
library(rattle)
library(rpart.plot)
library(RColorBrewer)

install.packages('RGtk2')
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')

fancyRpartPlot(prunetree2)
