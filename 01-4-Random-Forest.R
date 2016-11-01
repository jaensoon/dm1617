# Lab: Random Forest

# Importing Data ------------------------------------------------------------
# Data source is 
# https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)
# Some preprocessing of the original data set was done -> data/vehicle.csv

# library random forest
library(randomForest)
install.packages('randomForest')

vehicle <- read.table("/home/jwendling/Seafile/Study/Master/Semester 2/Data Mining/Praktikum/data-mining-lecture-assignments-201617ws/data/vehicle.dat", 
                   header=TRUE)
str(vehicle)
# Build Model ------------------------------------------------------------
?randomForest
rf_vehicle <- randomForest(formula=Class~., data=vehicle, ntree=500, mtry=4, do.trace=100)
print(rf_vehicle)
####
# FRAGE: Schauen Sie sich den Output von randomForest() an. Welche Informationen beinhaltet dieser?
# ANTWORT: Den Typ des RF, die Anzahl der Bäume und die Anzahl der Variablen pro Verzweigung
#
# FRAGE: Welches Argument steuert die “Anzahl Bäume” in randomForest() , welches Argument die “Anzahl Variablen an jedem Split”?
#        Was sind deren Default-Werte?
# ANTWORT: ntree - Anzahl der Bäume (Default: 500)
#          mtry - Anzahl der Variablen (Default: Wurzel von der Anzahl der Variablen --> Wurzel(19) = 4,358898944)
####

rf_vehicle$confusion
#Confusion matrix:
#      bus opel saab van class.error
#bus  214    1    0   3  0.01834862
#opel   2  103   98   9  0.51415094
#saab   6   73  123  15  0.43317972
#van    1    0    4 194  0.02512563

####
# FRAGE: Was wird schlecht klassifiziert?
# ANTWORT: 73 saab wurden als opel klassifiziert
#          98 opel wurden als saab klassifiziert
####

plot(rf_vehicle)
rf_vehicle$err.rate
plot(rf_vehicle$err.rate)

####
# FRAGE: Wie beurteilen Sie den Plot?
# ANTWORT: Die Fehlerraten sinken mit zunehmender Anzahl an Bäumen
####

tree_vehicle <- getTree(rf_vehicle, k=20, labelVar=TRUE)
tree_vehicle
?getTree

sum(tree_vehicle$status == -1)
# äquivalent zu
status_vehicle <- table(tree_vehicle$status)
status_vehicle[names(status_vehicle)==-1]
# äquivalten zu
treesize(rf_vehicle)[20]

####
# FRAGE: Wieviele Endpunkte hat sie?
# ANTWORT: 131 Endpunkte
####

plot(tree_vehicle)
?plot

####
# FRAGE: Zeichnen Sie den Baum, bis der erste Endpunkt erscheint — was für einen Endpunkt erhalten Sie?
# ANTWORT: 7
####

hist(treesize(rf_vehicle))

####
# FRAGE: Machen Sie dazu ein Histogramm. Wie beurteilen Sie dies?
# ANTWORT: Die häufigste Anzahl an Endpunkten ist im Bereich zwischen 130 und 140.
#          Die Verteilung entspricht einer Normalverteilung
####


#Aufgabe 4.3
rf_vehicle <- randomForest(formula=Class~., data=vehicle, ntree=500,
                           nodesize=1, mtry=14, maxnodes=10)
print(rf_vehicle)
rf_vehicle$err.rate
