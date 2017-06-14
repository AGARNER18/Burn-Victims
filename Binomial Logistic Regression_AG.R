# Amber Garner
# DATA 630
# Fall 2016
# Assignment 4
# Oct. 23, 2016
# Logistic Regression & Naive Bayes on burn datatset

#*********Setup*************************************** 

setwd("C:/Users/Amber/Desktop/Data Mining/Module 4")

# Install e1071 & arules for bayesian classification
install.packages("e1071")
install.packages("arules")
install.packages("dplyr")
install.packages("car")
library(e1071)
library(arules)
library(dplyr)
library(car)

# remove scientific notation
options(scipen=999)

# Load dataset
burn <- read.csv("burn.csv", header = T, sep = ",")

# Examine data
View(burn)
str(burn)
summary(burn)


#*********DATA PREPROCESSING*****************

# Remove ID
burn <- burn[,-1]

# Change names to make easier to work with
names(burn)<- c("facility", "death", "age", "gender", "race", "tbsa", "inh", "flame")

# Verify name changes
summary(burn)

# Corrrelation matrix of independent variables
corr <- cor(burn[,c(1,3:8)])
symnum(corr)
pairs(burn[,c(1,3:8)])

# Look at correlation coefficient for age, tbsa, inh, & flame
cor(burn[,c(2,5,6, 7)])

# Check for correlation between dependent and numeric independent variables
corr <- cor(burn[,c(2,3,6)])
corr
pairs(burn[,c(1,2,3,6)])

# Convert from numeric to factor type with accurate labels
# Convert to factor
# Death
burn$death <- factor(burn$death)
# Gender: female/male
burn$gender <- factor(burn$gender)
levels(burn$gender) <- list("female"=0, "male"=1)
# race: nonwhite/white
burn$race <- factor(burn$race)
levels(burn$race) <- list("non-white"=0, "white"=1)
# inhale: no/yes
burn$inh <- factor(burn$inh)
levels(burn$inh) <- list("no"=0, "yes"=1)
# flame: no/yes
burn$flame <- factor(burn$flame)
levels(burn$flame) <- list("no"=0, "yes"=1)

# Distribution of factor variables
summary(burn[,c(2,4,5,7,8)])

# Find percent of rows where gender is male
((nrow(burn[burn$gender=="male",]))/(nrow(burn)))* 100

# Find percent of rows where race is white
((nrow(burn[burn$race=="white",]))/(nrow(burn)))* 100

# Find percent of rows where inhalation is no
((nrow(burn[burn$inh=="no",]))/(nrow(burn)))* 100

# Find percent of rows where flame is yes
((nrow(burn[burn$flame=="yes",]))/(nrow(burn)))* 100

# Find percent of rows where death is 0-where patient lived
((nrow(burn[burn$death==0,]))/(nrow(burn)))* 100

# # Find percent of rows where death is 1-where patient died
((nrow(burn[burn$death==1,]))/(nrow(burn)))* 100

# Split into two groups to sample form alive group
alive <- burn[burn$death==0,]
dead <- burn[burn$death==1,]

# Get 200 random samples from alive
alive <- alive[sample(nrow(alive), 200), ]

# Combine alive and dead 
burn <- bind_rows(alive, dead)

# Clean up enviroment
rm(alive)
rm(dead)

# Check distribution after sampling
((nrow(burn[burn$death==0,]))/(nrow(burn)))* 100

# # Find percent of rows where death is 1-where patient died
((nrow(burn[burn$death==1,]))/(nrow(burn)))* 100

#*********LOGISTIC REGRESSION MODEL***********

# Divide data into training and test sets
# 70% into training, 30% into test
set.seed(1234)
ind <- sample(2, nrow(burn), replace = TRUE, prob = c(0.7, 0.3))
train.data <- burn [ind == 1, ]
test.data <- burn [ind == 2, ]

# Build general linear model with gaussian method 
model<-glm(death~., family="binomial", data=train.data)

# view model
print(model)

# Get details about model
summary(model)

# Verify that linear relationship exists between log odds and numeric variables
cor(train.data$age, model$fitted.values)
cor(train.data$tbsa, model$fitted.values)

# Check multicollinearity
vif(model)

# Intercept & coefficients for odds ratio
exp(coef(model))

# Get confidence intervals for model 
confint(model)

# Evaluate on training data
model$fitted.values[1:10]

# Round fitted values to nearest integer
# Find classification accuracy with confusion matrix
table(round(model$fitted.values), train.data$death)

# Calculate % correct alive predictions
(128/(128+15)) * 100

# Calculate % correct dead predictions
(92/(17+92)) * 100

# Calculate % correct for model
((128+92)/nrow(train.data)) * 100

# Calcualte % incorrect for model
((15+17)/nrow(train.data)) * 100

# Predict on test data
# Return first 10 rows
predict (model, test.data)[1:10]

# Predict on test data and round to nearest integer
# Store in new variable called mypredictions
mypredictions<-round(predict (model, test.data))

# Find accuracy of prediction on test data with confusion matrix
table(round(predict(model, newdata = test.data, type="response")), test.data$death)

# Calculate % correct alive predictions
(49/(49+6)) * 100

# Calculate % correct dead predictions
(37/(6+37)) * 100

# Calculate % correct for model
((49+37)/nrow(test.data)) * 100

# Calcualte % incorrect for model
((6+6)/nrow(test.data)) * 100

# Predicted vs residuals plot 
plot(predict(model),residuals(model), col=c("blue"))
lines(lowess(predict(model),residuals(model)), col=c("black"), lwd=2)
abline(h=0, col="grey")

#**********MINIMAL ADEQUATE MODEL********

# Minimal adequate model
model2 <- step(model)

# Details about model2
summary(model2)

# Check collinearity
vif(model2)

# Chi difference between null & residual
chidiff <- model2$null.deviance-model2$deviance

# degrees freedom between null & residual
dfdiff <- model2$df.null - model2$df.residual

# Get pchisquare value
chisq <- pchisq(chidiff, dfdiff)

# Calculate p-value from chi-squared
p <- 1-chisq
p

# Intercept & coefficients for odds ratio
exp(coef(model2))

# Get confidence intervals for model 
confint(model2)

# Round fitted values to nearest integer
# Find classification accuracy with confusion matrix
table(round(model2$fitted.values), train.data$death)

# Calculate % correct alive predictions
(130/(130+14)) * 100

# Calculate % correct dead predictions
(93/(15+93)) * 100

# Calculate % correct for model
((130+93)/nrow(train.data)) * 100

# Calcualte % incorrect for model
((15+14)/nrow(train.data)) * 100

# Create confusion matrix to find accuracy of new model
table(round(predict(model2, newdata = test.data, type="response")), test.data$death)

# Calculate % correct alive predictions
(50/(50+5)) * 100

# Calculate % correct dead predictions
(38/(38+5)) * 100

# Calculate % correct for model
((50+38)/nrow(test.data)) * 100

# Calcualte % incorrect for model
((5+5)/nrow(test.data)) * 100

# Predicted vs residuals plot 
plot(predict(model2), residuals(model2),col=c("blue"),main = "Residuals vs Predicted", ylab = "Residuals", xlab = "Predicted")
lines(lowess(predict(model2),residuals(model2)), col=c("black"), lwd=2)
abline(h=0, col="grey")


#***********DATA PREPROCESSING FOR NAIVE BAYES**************cor(train.data$age, model2$fitted.values)

burn$facility <- factor(burn$facility)
# death: alive/dead
levels(burn$death) <- list("alive"=0, "dead"=1) 
# age cut into 9 bins closed on right
burn$age <- cut(burn$age, breaks=c(0, 3, 10, 20, 30, 40, 50, 60, 70, 100), labels = c("Under 3","3-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"))
# gender: female/male
levels(burn$gender) <- list("female"=0,"male"=1)
# total burn surface area-frequency bins 7
burn$tbsa <- discretize(burn$tbsa, method="frequency", categories = 7)


# Verify that categories are accurate
summary(burn)

# divide data into training and test data sets
# Training is 70% & test is 30%
set.seed(1234)
ind <- sample(2, nrow(burn), replace = TRUE, prob = c(0.7, 0.3))
train.data <- burn[ind == 1, ]
test.data <- burn[ind == 2, ]

# Create naive bayes model
model3<-naiveBayes(death~., train.data)

# View model probabilities
print(model3)

# Create confusion matrix
table(predict(model3, train.data), train.data$death)

# Calculate % correct alive predictions
(129/(129+13)) * 100

# Calculate % correct dead predictions
(94/(16+94)) * 100

# Calculate % correct for model
((129+94)/nrow(train.data)) * 100

# Calcualte % incorrect for model
((16+13)/nrow(train.data)) * 100

# Create confusion matrix for test data
table(predict(model3, test.data), test.data$death)

# Calculate % correct alive predictions
(44/(44+5)) * 100

# Calculate % correct dead predictions
(38/(38+11)) * 100

# Calculate % correct for model
((44+38)/nrow(test.data)) * 100

# Calcualte % incorrect for model
((11+5)/nrow(test.data)) * 100

# Mosaic plot of model 3
mosaicplot(table(predict(model3, test.data), test.data$death), shade=TRUE, main="Predicted vs Observations")

#*********THE END**************