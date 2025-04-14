# Amanda Fox
# DATA 622 Assignment 3
# April 13, 2025

#---------------------------------
# Libraries
#---------------------------------

library(tidyverse)
library(caret)
library(MLmetrics)
library(kernlab)
library(ggplot2)
library(pROC)
library(doParallel)

#---------------------------------
# Load data from Assignment_2
#---------------------------------

# Import matrix of metrics for each algorithm tested in Assignment 2 for comparison
df_metrics <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_3/df_metrics.csv")
df_metrics

# Import balanced dataset and test/train split
df_bal <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_3/df_bal.csv")
df_bal_test <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_3/df_bal_test.csv")
df_bal_train <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_3/df_bal_train.csv")

# Factors: convert all char columns in train and test datasets to factors
df_bal_train <- df_bal_train %>%
  mutate(across(where(is.character), as.factor))

df_bal_test <- df_bal_test %>%
  mutate(across(where(is.character), as.factor))

# Explicitly set Y as a binary factor with controlled levels
# (second level is treated as "positive" in twoClassSummary)

df_bal_train$Y <- factor(df_bal_train$Y, levels = c("no", "yes"))
df_bal_test$Y  <- factor(df_bal_test$Y, levels = c("no", "yes"))

# STOP if factor levels do not match between train and test
stopifnot(all(names(df_bal_train) == names(df_bal_test)))

glimpse(df_bal_train)
glimpse(df_bal_test)

#---------------------------------
# Prepare to store metrics
#---------------------------------

# Initialize new matrix to store new metrics 
matrix_metrics <- matrix(NA, nrow = 0, ncol = 6) 
colnames(matrix_metrics) <- c("Model",
                              "Accuracy", 
                              "Precision", 
                              "Recall", 
                              "F1", 
                              "AUC")


#----------------------------------------
# Start parallel processing
#----------------------------------------

# Stop if one is running from last session
if (exists("cl")) {
  try(stopCluster(cl), silent = TRUE)
  rm(cl)
}

# Start
num_cores <- detectCores() - 1 # Use one less core
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

#----------------------------------------
# Scale and center data
#----------------------------------------

# SVMs are sensitive to the scale of features, so we use the training set to compute 
# the mean and standard deviation of each numeric variable (preProcess).
# We then apply this transformation to both the training and test sets.
# This ensures consistent scaling and avoids data leakage from the test set.

preproc <- preProcess(df_bal_train, 
                      method = c("center", "scale"))
train_svm <- predict(preproc, df_bal_train)
test_svm <- predict(preproc, df_bal_test)


#-------------------------------
# Train and test: SVM Linear Kernel
#-------------------------------

set.seed(123)

# linear kernel: linear boundaries, higher bias, lower variance
# five fold cross validation, PRECISION as metric

# Using the caret package, we train an SVM with a linear decision boundary
# and 5-fold cross-validation and tune the cost parameter (C). Smaller C values result in simpler boundaries 
# with higher bias, while larger values allow more complex fits with higher variance.
# We evaluate model performance using ROC AUC and choose the best model based on this metric.


# train model
svm_linear <- train(
  Y ~ ., 
  data = train_svm, 
  method = "svmLinear",
  trControl = trainControl(
    method = "cv", 
    number = 5,
    classProbs = TRUE,
    summaryFunction = prSummary
  ),
  metric = "Precision",
  tuneGrid = expand.grid(C = c(0.1, 1, 10))
)

# Keep only rows in test set where Loan_Profile exists in training set
test_svm <- test_svm[test_svm$Loan_Profile %in% levels(train_svm$Loan_Profile), ]

# Drop unused factor levels to avoid warnings
test_svm$Loan_Profile <- droplevels(test_svm$Loan_Profile)


# Predict on test set
pred_svm_linear <- predict(svm_linear, newdata = test_svm)
prob_svm_linear <- predict(svm_linear, newdata = test_svm, type = "prob")[, 2]

# Confusion matrix and AUC
cm_svm_linear <- confusionMatrix(pred_svm_linear, test_svm$Y)
auc_svm_linear <- auc(roc(test_svm$Y, prob_svm_linear))

# add metrics to the matrix 
matrix_metrics <- rbind(matrix_metrics, 
                        c("8 SVM Linear",
                          cm_svm_linear$overall["Accuracy"],
                          cm_svm_linear$byClass["Precision"],
                          cm_svm_linear$byClass["Recall"],
                          cm_svm_linear$byClass["F1"],
                          auc_svm_linear))

matrix_metrics

# convert to df and change to numeric values
df_svm_metrics <- as.data.frame(matrix_metrics)

df_svm_metrics <- df_svm_metrics %>%
  mutate(
    Accuracy = as.numeric(Accuracy),
    Precision = as.numeric(Precision),
    Recall = as.numeric(Recall),
    F1 = as.numeric(F1),
    AUC = as.numeric(AUC)
  )

# create all metrics table
df_all_metrics <- bind_rows(df_metrics, df_svm_metrics)
df_all_metrics

#write_csv(df_all_metrics,"df_all_metrics_temp.csv")


#-------------------------------
# Train and test: SVM RBF Kernel
#-------------------------------

# radial kernel, five fold cross validation, Precision as metric

# Set the same seed for all SVM models to ensure consistent cross-validation folds
# This allows fair comparison between linear and radial kernels using the same data splits

set.seed(123)

# estimate sigma values: sigma = 1/(2*gamma) in caret
sigest(Y ~ ., data = train_svm)
sigest

# train model
#svm_radial <- train(
#  Y ~ ., 
#  data = train_svm, 
#  method = "svmRadial",
#  trControl = trainControl(
#    method = "cv", 
#    number = 5,
#    classProbs = TRUE,
#    summaryFunction = prSummary
#  ),
#  metric = "Precision",
#  tuneGrid = expand.grid(
#    C = c(1, 10),
#    sigma = c(0.05, 0.09)
#  )
#)

svm_radial <- train(
  Y ~ ., 
  data = train_svm, 
  method = "svmRadial",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = prSummary,
    search = "random",
    verboseIter = TRUE
  ),
  metric = "Precision",
  tuneLength = 4
)

# add metrics to the matrix 

matrix_metrics <- rbind(matrix_metrics, 
                        c("9 SVM Radial",
                          cm_svm_radial$overall["Accuracy"],
                          cm_svm_radial$byClass["Precision"],
                          cm_svm_radial$byClass["Recall"],
                          cm_svm_radial$byClass["F1"],
                          auc_svm_radial))

# convert to df and change to numeric values
df_svm_metrics <- as.data.frame(matrix_metrics)

df_svm_metrics <- df_svm_metrics %>%
  mutate(
    Accuracy = as.numeric(Accuracy),
    Precision = as.numeric(Precision),
    Recall = as.numeric(Recall),
    F1 = as.numeric(F1),
    AUC = as.numeric(AUC)
  )

# create all metrics table
df_all_metrics <- bind_rows(df_metrics, df_svm_metrics)
df_all_metrics


#---------------------------------
# Compare all models and make bar chart
#---------------------------------

df_all_metrics


# Pivot for ggplot
df_long <- df_all_metrics %>% 
  pivot_longer(cols = c(-Model),
               names_to = "Metric")

# Create comparison bar chart
plot_compare <- df_long %>%  
  ggplot(aes(x = reorder(Model, desc(Model)),
             y = value,
             fill = Metric)) +
  geom_bar(stat = "identity", 
           show.legend = FALSE) +
  geom_text(aes(label = round(value, 3)), 
            hjust = -0.1, size = 2.5) +
  facet_wrap(~ Metric, 
             scales = "free_y",
             ncol = 1) +
  labs(title = "Model Performance by Metric",
       x = "",
       y = "")+
  coord_flip() +
  theme_minimal()
plot_compare  


#---------------------------------
# parallel processing stop 
#---------------------------------
stopCluster(cl)
registerDoSEQ()



  # check for error: test set has unrecognized factor not in train
  
levels(train_svm$Loan_Profile)
levels(test_svm$Loan_Profile)

str(train_svm)
str(test_svm)




