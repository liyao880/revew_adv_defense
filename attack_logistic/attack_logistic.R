# Attack Logistic Regression Model with FGSM and PGD attack
#setwd("")

## 1. Load packages
library(dslabs)
library(LiblineaR)
library(mclust)
source('utility.R')

## 2. Load the handwritten digits data
mnist = read_mnist()
#----To do binary classification, we only keep the images with label 0 and 1.----#
train = get_zero_one(mnist$train)
test = get_zero_one(mnist$test)

## 3. Train a Logistic Classifier
#----Logistic Regression----#
classifier = LiblineaR(train$images, train$labels, type=6, cost=0.5, bias=0)
cat("Number of NA in W:", sum(is.na(classifier$W)))
prediction = predict(classifier, test$images, proba = FALSE, decisionValues = FALSE)$predictions
prerror=classError(prediction,test$labels)
cat("The prediciton accuracy:", 1-prerror$errorRate)

## 4. Peform PGD attack on test set
epsilon=8 
alpha=1
iters=40 # When iters=1, it is FGSM attack
attack = attack_over_test(test,prediction,classifier,epsilon,alpha,iters)

## 5. Visualization of adversarial examples
print("You can select from the following indices to visualize successful attack:")
print(attack)
Visual_adv(i=2088,test,prediction,classifier,epsilon,alpha,iters)
