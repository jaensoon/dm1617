# Lab: Association Rules for Supermarket Transaction

# Libraries --------------------------------------------------------------------
library(arules)
install.packages('arules')

# Load Data --------------------------------------------------------------------
# Example: Identifying Frequently-Purchased Items

# Load the supermarket data into a sparse matrix
# Since there are 169 different items in the supermarket data, the sparse 
# matrix will contain 169 columns. With the relatively small transactional 
# dataset used here (9865 transactions and 169 items),the matrix contains 
# nearly 1.662.115 million cells, most of which contain zeros (hence, the name 
# "sparse" matrix—there are very few nonzero values). 
# There is no benefit to storing all these zero values, a sparse matrix does not 
# actually store the full matrix in memory; it only stores the cells that are 
# occupied by an item. This allows the structure to be more memory efficient than
# an equivalently sized matrix or data frame.
transactions <- read.transactions("/home/jwendling/Seafile/Study/Master/Semester 2/Data Mining/Lab/dm1617/data/supermarket_transaction_baskets.csv", sep = ",")

# Explorative Analytics --------------------------------------------------------
# look at the first five transactions
inspect(transactions[1:5])

# Summary
summary(transactions)
inspect(transactions)
show(transactions)
transactionInfo(transactions)
?transactions

# How many items where purchased during the store's 30 days of operation:
# Your answer: 43367
# 
# How many items does a transaction contain in average?
# Your answer:
# 43367 / 9835 = 4,409456024
# Anzahl der Gegenstände durch die Anzahl der Transaktionen
# Auch in der Summary zu sehen


# examine the frequency of the first three items
# items are stored in the matrix in alphabetical order 
itemFrequency(transactions)

# plot the frequency of items
# with support 0.1
itemFrequencyPlot(transactions, support = 0.10)

#topN
itemFrequencyPlot(transactions, topN = 10)

# visualize the sparse matrix for the first 20 transactions
image(transactions[1:20])

# visualize of a random sample of 100 transactions
image(sample(transactions, 100))

# Models -----------------------------------------------------------------------

# default settings result in ???  rules learned
?apriori

apriori(transactions)

# What are default values for support and confidence?
# support: 0.1
# confidence: 0.8
# 
# How do you interpret support of 0.1?
# Your answer: Das bei allen Itemmengen der Support mindestens 10% sein muss, damit für diese Regeln erstellt werden.
#              Mit der Default-Einstellung wurden keine Regeln erstellt!

# Set better support and confidence levels to learn more rules
# You are interested in items which are purchased twice a day during 30 days.
# This leads to support of ???
# Your answer: 30 Tage a 2 Gegenstände = 60
#              Diese wird durch die Gesamtanzahl der Transaktion 9835 geteilt
#              Das ergibt einen Support von 0,006100661
#
# Start with a confidence threshold of 0.25
# Set minlen = 2 as we are only interested in in rules with two or more items

#inspect(head(sort(rules,by="lift"), n=20))
supermarketrules <- apriori(transactions,parameter=list(supp=0.006100661, conf=0.25, minlen=2, maxlen=10, target='rules'))
supermarketrules

# Evaluating model performance -------------------------------------------------
# summary of supermarket association rules
summary(supermarketrules)

# look at the first three rules
inspect(supermarketrules[1:3])

## Step 5: Improving model performance ----

# sorting supermarket rules by lift
inspect(sort(supermarketrules, by="lift")[1:5])

# finding subsets of rules containing any berry items
berryrules <- subset(supermarketrules, items %in% "berries")
inspect(berryrules)

# writing the rules to a CSV file
write(supermarketrules, file = "/home/jwendling/Seafile/Study/Master/Semester 2/Data Mining/Lab/dm1617/04/supermarketrules.csv",
      sep = ",", quote = TRUE, row.names = FALSE)

# converting the rule set to a data frame
supermarketrules_df <- as(supermarketrules, "data.frame")
str(supermarketrules_df)

# Interpret the rules
# Your summary:
# Ziel war es für Produkte, die zwei mal täglich gekauft werden, Regeln zu erstellen. Die Regeln sind sehr gut, denn bei allen, außer bei einer ist
# der Lift über 1.0. Die Außnahme ist {bottled beer} => {whole milk} mit einem Lift von 0.99.
