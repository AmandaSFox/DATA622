# Amanda Fox
# DATA 622 Assignment 1
# March 2, 2025

#---------------------------------
# Libraries
#---------------------------------

library(tidyverse)
library(ggplot2)
library(purrr)
library(knitr)
library(scales)
library(reshape2)
library(GGally)
library(RColorBrewer)

#---------------------------------
# Load Data
#---------------------------------

#url <- "https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_1/bank-full.csv"

# NEW - WITH ADDITIONAL FIELDS 
url <- "https://raw.githubusercontent.com/AmandaSFox/DATA622/refs/heads/main/Assignment_1/bank%2Bmarketing/bank-additional/bank-additional/bank-additional-full.csv"

df <- read_delim(url,
                 delim = ";",
                 quote = "\"")

#---------------------------------
# Overview
#---------------------------------

glimpse(df)

# missing data
colSums(is.na(df))

# duplicate records
nrow(df) - nrow(distinct(df))

# change col names based on documentation (REVISED)
df_final <- df %>% 
  set_names("Age","Occupation","Marital_Status","Education",
            "In_Default","Housing_Loan","Personal_Loan",
            "Contact_Type","Month","Day","Duration","Contacts_This_Campaign",
            "Days_Since_Last_Campaign","Contacts_Before_This_Campaign",
            "Previous_Outcome",
            "emp.var.rate","cpi","cci","euribor3m","nr.employed",
            "Y")

glimpse(df_final)

str(df_final)

# new - group up some new education levels (not an issue in old dataset)

df_preprocessed <- df_final %>%
  mutate(Education2 = case_when(
    grepl("^basic|^illiterate", 
          Education, 
          ignore.case = TRUE) ~ "less.than.hs",
    TRUE ~ as.character(Education))) 

df_preprocessed <- data.frame(df_preprocessed)
str(df_preprocessed)

df_preprocessed %>% 
  count(Education2) 
df_preprocessed %>% 
  count(Occupation) 

# occupation/education feature REMOVED AT END - TOO MANY VALUES FOR MODELING

df_preprocessed <- df_preprocessed %>%
  mutate(Occupation_Education = paste(Occupation, Education2, sep = "_"))
df_preprocessed %>% 
  count(Occupation_Education) 

# loan profile feature: default, housing, personal loans
df_preprocessed <- df_preprocessed %>%
  mutate(Loan_Profile = case_when(
    Housing_Loan == "yes" & Personal_Loan == "yes" & In_Default == "yes" ~ "Both Loans - In Default",
    Housing_Loan == "yes" & Personal_Loan == "yes" & In_Default == "no" ~ "Both Loans - No Default",
    Housing_Loan == "yes" & Personal_Loan == "no" & In_Default == "yes" ~ "Housing Loan - In Default",
    Housing_Loan == "yes" & Personal_Loan == "no" & In_Default == "no" ~ "Housing Loan - No Default",
    Housing_Loan == "no" & Personal_Loan == "yes" & In_Default == "yes" ~ "Personal Loan - In Default",
    Housing_Loan == "no" & Personal_Loan == "yes" & In_Default == "no" ~ "Personal Loan - No Default",
    Housing_Loan == "no" & Personal_Loan == "no" & In_Default == "yes" ~ "No Loans - In Default",
    TRUE ~ "No Loans - No Default"
  ))


df_preprocessed %>%
  count(Loan_Profile, sort = TRUE) %>%
  mutate(Pct = round((n / sum(n)) * 100, 1))

glimpse(df_preprocessed)

# Remove unneeded columns
df_preprocessed <- df_preprocessed %>%
  select(-Housing_Loan, 
         -Personal_Loan, 
         -In_Default,
         -Education,
         -Occupation_Education)

df_preprocessed %>% 
  write_csv("df_preprocessed.csv")
