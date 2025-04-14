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

###############################################################

##############################################
#FIGURE OUT RBF ERRORS
##############################################

# Set the same seed for all SVM models to ensure consistent cross-validation folds
set.seed(123)

# estimate sigma values: sigma = 1/(2*gamma) in caret
sigest(Y ~ ., data = train_svm)

set.seed(123)

# Train on median value for sigma and two values for C due to memory issues
svm_radial <- train(
  Y ~ ., 
  data = train_svm, 
  method = "svmRadial",
  trControl = trainControl(
    method = "cv", 
    number = 5,
    classProbs = TRUE,
    summaryFunction = prSummary,
    verboseIter = TRUE
  ),
  metric = "Precision",
  tuneGrid = expand.grid(
    C = c(1, 10),
    sigma = c(0.05139891)
  )
)


###############################
# Train model RANDOM
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
  tuneLength = 3
)

save(svm_radial, file = "prob_svm_radial.RData")

# Keep only rows in test set where Loan_Profile exists in training set
test_svm <- test_svm[test_svm$Loan_Profile %in% levels(train_svm$Loan_Profile), ]

# Drop unused factor levels to avoid warnings
test_svm$Loan_Profile <- droplevels(test_svm$Loan_Profile)

# Predict on test set
pred_svm_radial <- predict(svm_radial, newdata = test_svm)
prob_svm_radial <- predict(svm_radial, newdata = test_svm, type = "prob")[, 2]

# Confusion matrix and AUC
cm_svm_radial <- confusionMatrix(pred_svm_radial, test_svm$Y)
auc_svm_radial <- auc(roc(test_svm$Y, prob_svm_radial))

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


df_svm_metrics

# Import SVM metrics from earlier sessions
df_svm_metrics <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_3/df_svm_metrics_new.csv")
df_svm_metrics

# Combine with metrics from Assignment 2 to create all metrics table
df_all_metrics <- bind_rows(df_metrics, df_svm_metrics)
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

## Cleanup

stopCluster(cl)
registerDoSEQ()

######################################################

  # check for error: test set has unrecognized factor not in train
  
levels(train_svm$Loan_Profile)
levels(test_svm$Loan_Profile)

str(train_svm)
str(test_svm)




