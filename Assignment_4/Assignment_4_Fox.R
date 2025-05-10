# Assignment 4
# Predicting Churn in Telecommunications

#-----------------------------------
# Libraries
#-----------------------------------

library(tidyverse)
library(fastDummies)
library(ggplot2)
library(summarytools)
library(scales)

#-----------------------------------
# Data Prep
#-----------------------------------

# Load data
df_raw <- read_csv("https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_4/WA_Fn-UseC_-Telco-Customer-Churn.csv")

glimpse(df_raw)

# Missing values: 
colSums(is.na(df_raw)) #11 missing TotalCharges

missing_chgs <- df_raw %>% 
  filter(is.na(TotalCharges)) #these records have tenure = 0 and no charges
missing_chgs

df_raw <- df_raw %>%
  filter(!is.na(TotalCharges)) #removed records: too new for churn model

# Duplicate rows
sum(duplicated(df_raw)) # no duplicate rows found

#--------------------------------
# Prepare Data: Fix Data Types for Modeling
#--------------------------------

# Standardize flags to numeric 0/1: 
# OK for trees and necessary for neural networks

# List of flag cols with char type and Yes/No values
yesno_cols <- c("Partner", "Dependents", "PhoneService", "MultipleLines",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
                "PaperlessBilling", "Churn")  

# For paper: Original values in cols
#Table 1. Value Counts for Original Yes/No Service Flags
#(Derived from df_raw; includes values like “Yes”, “No”, and “No internet service”)

yesno_counts_before <- df_raw %>%
  select(all_of(yesno_cols)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  count(variable, value) %>%
  arrange(variable, desc(n))

yesno_counts_before

# Manual one-hot encoding:
# create new flag cols for "Phone Svc" and "Internet Svc" using values in existing cols

df_model <- df_raw %>%
  # create binary flag for phone service
  Phone_Svc = if_else(PhoneService == "Yes", 1, 0) %>% 
  # create binary flags for internet service and type
  mutate(Internet_Svc = if_else(InternetService == "No", 0, 1),
         DSL    = if_else(InternetService == "DSL", 1, 0),
         Fiber  = if_else(InternetService == "Fiber optic", 1, 0))

# verify new cols
df_model %>% 
  count(InternetService, Internet_Svc, DSL,Fiber)
df_model %>% 
  count(PhoneService, Phone_Svc)

# Recheck cols with internet and phone service flags 
df_model %>%
  count(Internet_Svc,OnlineSecurity) %>%
  arrange(OnlineSecurity)

df_model %>%
  count(Phone_Svc,MultipleLines) %>%
  arrange(MultipleLines)

# One-Hot encoding using package (gender, contract, and payment types)

# gender
df_model %>% 
  count(gender)

# contract
df_model %>%
  count(Contract)

# payment type
df_model %>%
  count(PaymentMethod)

df_model <- df_model %>%
  dummy_cols(
    select_columns = c("PaymentMethod", "Contract", "gender"),
    remove_selected_columns = TRUE
  )

glimpse(df_model)

# Clean up all binary flags to 0/1 (treat "No service" as 0)
df_model <- df_model %>% 
    mutate(MultipleLines = if_else(MultipleLines == "Yes", 1, 0),
           OnlineSecurity = if_else(OnlineSecurity == "Yes", 1, 0),
           OnlineBackup = if_else(OnlineBackup == "Yes", 1, 0),
           DeviceProtection = if_else(DeviceProtection == "Yes", 1, 0),
           TechSupport = if_else(TechSupport == "Yes", 1, 0),
           StreamingTV = if_else(StreamingTV == "Yes", 1, 0),
           StreamingMovies = if_else(StreamingMovies == "Yes", 1, 0),
           Partner = if_else(Partner == "Yes", 1, 0),
           Dependents = if_else(Dependents == "Yes", 1, 0),
           PaperlessBilling = if_else(PaperlessBilling == "Yes", 1, 0),
           Churn = if_else(Churn == "Yes", 1, 0))

glimpse(df_model)

# For Paper: recheck values in flag cols
#Table 2. Value Counts After Binary Conversion of Service Flags
#(Derived from df_model; all flags shown as 0 or 1, including Phone_Svc and Internet_Svc)

yesno_cols_after <- c("Partner", "Dependents", "MultipleLines",
                      "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies",
                      "PaperlessBilling", "Churn","Phone_Svc", "Internet_Svc",
                      "DSL","Fiber")

yesno_counts_after <- df_model %>%
  select(all_of(yesno_cols_after)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  count(variable, value) %>%
  arrange(variable, desc(n))

# Recheck cols with internet and phone service flags 
df_model %>%
  count(OnlineSecurity, Internet_Svc) %>%
  arrange(OnlineSecurity)

df_model %>%
  count(MultipleLines, Phone_Svc) %>%
  arrange(MultipleLines)

