# DATA SCIENCE WITH R PROJECT
# 1. Import necessary libraries to the workspace for use
library(csvread)    #For csv files
library(plyr)       #For changing features in a dataset
library(ggplot2)    #For advanced plotting in r.
library(caret)      #For confusion matrices etc.
library(e1071)      #For general machine learning algorithms
install.packages("glm2")
library(glm2)       #For Logistic regression
library(rpart)      #For decision trees


# 2.  Set the working directory and load the dataset
setwd(choose.dir())

# 3. Load the Dataset
student_data <- read.csv("Project 1_Dataset.csv")

# 4. View the dataset
View(student_data)
# Dataset has 400 observations with 7 variables.

# 5. Check the Structure of the dataset to understand the variable types.
str(student_data)
# Notice the admit variable which is our response variable is an integer type.

# 6. Convert "admit" response variable from integer to factor 
student_data$admit <- sapply(student_data$admit, factor)
# Check the structure of the dataset to see the change.

# 7. Check the summary to understand the statistical features of the dataset
summary(student_data)
# We can see 273 not admitted and 127 admitted

# 8. Lets try to view some statistics on the dataset 
# Bin the data by GRE
student_data_binned <- transform(student_data, GREBin = ifelse(gre<440, "LOW", 
                                                               ifelse(gre<580, "MEDIUM", "HIGH")))
# Bin by GPA
student_data_binned <- transform(student_data_binned, GPABin = ifelse(gpa<3.000, "LOW",
                                                                      ifelse(gpa<3.500, "MEDIUM", "HIGH")))
# View the binning
View(student_data_binned)

# create a 2 way frequency table to try to get insights into the gre bin and the admit status also gpa bin and admit.
table(student_data_binned$GREBin, student_data_binned$admit)
table(student_data_binned$GPABin, student_data_binned$admit)

# Lets plot a histogram of the GPA and GRE distribution and gender.
hist(student_data_binned$gpa, breaks = 3, col = "red")
hist(student_data_binned$gre, breaks = 3, col = "darkgreen")


# 9. Test the logistic regression model
model <- glm(admit ~ ., family = binomial(link = "logit"), data = student_data)
# Check the summary of the model to determine useful variables
summary(model)
# We can see that the ses, gender, and race features are insignificant to the model
# We can then safely drop these columns.

# Run an Anova to analyze the table of variance
anova(model, test="Chisq")
# create a prediction of the whole dataset. 
glm_probs = predict(model, type = "response")
# Check the top 5 rows to see prediction
glm_probs[1:5]

# Set the prediction threshold to test the accuracy of the model
glm_pred <- ifelse(glm_probs > 0.551, "1", "0")
# Use a confusion matrix to test accuracy of model.
confusionMatrix(student_data$admit, glm_pred, positive = "1")
# We can see an accuracy of 70% on the model using all the features of the dataset.
# We can adjust the threshold to see if the prediction will be better but so far, 0.551 produces the best accuracy.
# So lets remove the irrelevant columns in our model and test again to see if it performs better

# Delete irrelevant columns
del_vars = names(student_data) %in% c("ses", "Gender_Male", "Race")
student_data_clean = student_data[!del_vars]
# View the new dataset
View(student_data_clean)


# 10. Split the dataset into train and test to use in our model
# Randomize the selection first
sample_split <- floor(.8 * nrow(student_data_clean))
set.seed(1)
train <- sample(seq_len(nrow(student_data_clean)), size = sample_split)

# Create the train and test sets
admit_train <- student_data_clean[train, ]
admit_test <- student_data_clean[-train, ]


# 11. Run the model again
log_model <- glm(admit ~ ., family = binomial(link = "logit"), data = admit_train)
# Predict the test setset
Prediction <- predict(log_model, newdata = subset(admit_test, select = c("gre", "gpa", "rank")))
# Set the prediction threshold
Prediction_results <- ifelse(Prediction > 0.4, 1, 0)
# Setting the threshold for this model has to be done over and over to get a rather balanced point.
# So far best seems to be 0.4 threshold to avoid overfitting the model

# Create a confusion Matrix to see the accuracy of the model
confusionMatrix(admit_test$admit, Prediction_results, positive = "1")
# Accuracy then becomes 72.5%
summary(log_model)

# Just another confusionMatrix
Pred <- table(pred= Prediction_results, true = admit_test$admit)
print(Pred)

# 12. Trying other Modelling techniques
# a. Support Vector Machines
svm_model <- svm(admit~., admit_train)
# Predict the svm model
Prediction2 <- predict(svm_model, newdata = subset(admit_test, select = c("gre", "gpa", "rank")))
confusionMatrix(admit_test$admit, Prediction2, positive = "1")
# Accuracy is at 71.25%
summary(svm_model)


# b. Decision Tree
tree_model <- rpart(admit~., data = admit_train, method = "class")
tree_model     
printcp(tree_model)
summary(tree_model)
plotcp(tree_model)
plot(tree_model)
# Predict the decision tree
Prediction3 <- predict(tree_model,newdata = subset(admit_test, select = c("gre", "gpa", "rank")),type="class")
confusionMatrix(admit_test$admit, Prediction3, positive = "1")
# Performs much better at predicting on the test dataset
# An accuracy of 65% seems better than the other models.

# 13. Getting the prediction probabilities from the decision tree model.
Predict_proba <- predict(tree_model,newdata = subset(admit_test, select = c("gre", "gpa", "rank")),type="prob")
# Attaching these probabilities to the test set
admit_test$proba <- Predict_proba
View(admit_test)
str(admit_test)

# 14. Plotting the gpa against the probabilities.
p2 <- ggplot(admit_test, aes(gpa, proba)) + geom_point() + scale_x_log10() + scale_y_log10()
plot(p2)
