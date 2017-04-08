# Set working directory
setwd("/Users/Animesh/SortShit/NEU/Courses/CS 6140 - Machine Learning/Project")

# Importing the data set
dataset = read.csv('data/cleaned_all.csv')

split_factor = split(dataset, dataset$Validation)
training_set = split_factor$False
test_set = split_factor$True
rm(split_factor)

# Splitting dataset into training and test set based on validation column
# install.packages('caTools')
library(caTools)

training_set = training_set[,1:1347]
test_set = test_set[,1:1347]

# Actual response from the test set
actual_response = test_set[,1347:1347]

# Baseline model - predict the mean of the training data
base_mean = mean(training_set$GNI.per.capita..constant.2005.US.._next)

RMSE_baseline = sqrt(mean((base_mean - actual_response)^2, na.rm = TRUE))
MAE_baseline = mean(abs(base_mean - actual_response), na.rm = TRUE)

# Decision tree regression

# USING RPART
# Fitting decision tree regression to the dataset
# install.packages("rpart") 
library(rpart)

# Tree with default settings, goal is to 
# check cross validation results to determine minimum
# cross validation error and prune the tree using new 
# complexity factor

# Grow tree
fit = rpart(GNI.per.capita..constant.2005.US.._next ~ .,
            method = "anova", # anova for regression
            data = training_set)

# Predicting response using fit and test_set
prediction = predict(fit, newdata = test_set)

# Named vector to numeric only vector
prediction_numeric = unname(prediction)

# Root mean squared error
RMSE_fulltree <- sqrt(mean((prediction_numeric - actual_response)^2, na.rm = TRUE))

# Mean absolute error
MAE_fulltree = mean(abs(prediction_numeric - actual_response), na.rm = TRUE)
  
# Checking cross-validation results (xerror column)
printcp(fit)

# Finding complexity factor for minimum xerror
min_cp <- fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]

# Prune the full tree
pruned_tree = prune(fit,cp = min_cp)

# Predicting response using pruned tree fit and test_set
prediction_pruned = predict(pruned_tree, newdata = test_set)

# Named vector to numeric only vector
prediction_pruned_numeric = unname(prediction_pruned)

# Root mean squared error
RMSE_pruned <- sqrt(mean((prediction_pruned_numeric - actual_response)^2, na.rm = TRUE))

# Mean absolute error
MAE_pruned = mean(abs(prediction_pruned_numeric - actual_response), na.rm = TRUE)

# Print the cross validation results for the pruned tree
printcp(pruned_tree)


# No difference whatsoever was observed with pruning
# Now, applying parameter tuning in an attempt to reduce RMSE and MAE
# after lots of manual runs, witha a cp of 10^-4 and
# min. observations in terminal node of 10, we were able to get very low
# RMSE and MAE (almost 1/2 of the previous values)

# Grow tree
fit_tuned = rpart(GNI.per.capita..constant.2005.US.._next ~ .,
            method = "anova", # anova for regression
            data = training_set,
            control = rpart.control(minbucket = 5, cp = 0.0001))

printcp(fit_tuned)

# Predicting response using fit and test_set
prediction_tuned = predict(fit_tuned, newdata = test_set)

# Named vector to numeric only vector
prediction_numeric_tuned = unname(prediction_tuned)

# Root mean squared error
RMSE_fulltree_tuned <- sqrt(mean((prediction_numeric_tuned - actual_response)^2, na.rm = TRUE))

# Mean absolute error
MAE_fulltree_tuned = mean(abs(prediction_numeric_tuned - actual_response), na.rm = TRUE)


################################################################
################# Random Forest Regression #####################
################################################################


# Importing the imputed data set using kNN
# randomForest package handles NA values poorly,
# so we are using the imputed dataset for this

dataset_imputed = read.csv('data/imputed-data.csv')

split_factor = split(dataset_imputed, dataset_imputed$Validation)
training_set_imputed = split_factor$False
test_set_imputed = split_factor$True
rm(split_factor)


training_set_imputed = training_set_imputed[,1:1329]
test_set_imputed = test_set_imputed[,1:1329]

# Convert strings (Country name) to numeric values
training_set_imputed$CountryName = as.numeric(training_set_imputed$CountryName)
test_set_imputed$CountryName = as.numeric(test_set_imputed$CountryName)

# Real test set response
actual_response_imputed = test_set_imputed$GNI.per.capita..constant.2005.US.._next

library(randomForest)

# Create a random forest
fit_rf = randomForest(GNI.per.capita..constant.2005.US.._next ~ .,
                      data = training_set_imputed, 
                      importance = TRUE, 
                      na.action = na.roughfix)

# Using the importance()  function to calculate the importance of each variable
imp_predictors <- as.data.frame(sort(importance(fit_rf)[,1],decreasing = TRUE),optional = T)

# Get the best and worst predictors
head(imp_predictors)

# Perform prediction
prediction_rf = predict(fit_rf,test_set_imputed)

prediction_rf_numeric = unname(prediction_rf)

# RMSE and MAE for Random forests 
RMSE_rf = sqrt(mean((prediction_rf_numeric - actual_response_imputed)^2))
MAE_rf = mean(abs(prediction_rf_numeric - actual_response_imputed))

RMSE_rf
MAE_rf

# Final analysis

# Create a data frame with the error metrics for each method
accuracy = 
  data.frame(Method = 
            c("Baseline","Full tree","Tuned Full Tree","Random forest"),
            RMSE = c(RMSE_baseline,RMSE_fulltree,RMSE_fulltree_tuned,RMSE_rf),
            MAE = c(MAE_baseline,MAE_fulltree,MAE_fulltree_tuned,MAE_rf))

# Round the values and print the table
accuracy$RMSE = round(accuracy$RMSE,2)
accuracy$MAE = round(accuracy$MAE,2) 

accuracy

# Create a data frame with the predictions for each method
prediction_all = data.frame(actual = actual_response,
                            baseline = base_mean,
                            full_tree = prediction,
                            tuned_tree = prediction_tuned,
                            random_forest = prediction_rf)

head(prediction_all)

library(tidyr)

prediction_all = gather(prediction_all,key = model,value = predictions,2:5)

library(ggplot2)

# Predicted vs. actual for each model
ggplot(data = prediction_all, aes(x = actual, y = predictions)) +
  geom_point(colour = "blue") +
  geom_abline(intercept = 0, slope = 1, colour = "red") +
  geom_vline(xintercept = 23, colour = "green", linetype = "dashed") +
  facet_wrap(~ model,ncol = 2) +
  ggtitle("Predicted vs. Actual, by model")


# Results summary

# Full tree
printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# create additional plots 
rsq.rpart(fit) # visualize cross-validation results  	

# create attractive postcript plot of tree 
post(fit, 
     file = "/Users/Animesh/SortShit/NEU/Courses/CS 6140 - Machine Learning/Project/Results/Full Tree/tree.ps",  
     title = "Regression Tree")

# Tuned tree
printcp(fit_tuned) # display the results 
plotcp(fit_tuned) # visualize cross-validation results 
summary(fit_tuned) # detailed summary of splits

# create additional plots 
rsq.rpart(fit_tuned) # visualize cross-validation results  	

# plot tree 
plot(fit_tuned, uniform=TRUE, 
     main="Regression Tree for GNI")
text(fit_tuned, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postcript plot of tree 
post(fit_tuned, 
     file = "/Users/Animesh/SortShit/NEU/Courses/CS 6140 - Machine Learning/Project/Results/Tuned Tree/tree.ps",  
     title = "Regression Tree")
