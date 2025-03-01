library(tidyverse)
library(ggplot2)
library(purrr)
library(reshape2)
#library(corrplot)
library(GGally)


# prep data for charts 
df_final <- df %>% 
  set_names("Age","Occupation","Marital_Status","Education",
            "In_Default","Avg_Balance","Housing_Loan","Personal_Loan",
            "Contact_Type","Day","Month","Duration","Contacts_This_Campaign",
            "Days_Since_Last_Campaign","Contacts_Before_This_Campaign",
            "Previous_Outcome","Y")

# prep for charts: removed "Day" as it is not meaningful numerically
df_num <- df_final %>% 
  select(-Day) %>% 
  select(where(is.numeric))
df_long <- df_num %>% 
  pivot_longer(everything())

# histogram for distributions
plot_hist <- df_long %>% 
  ggplot(aes(value))+
    geom_histogram(bins = 30,fill="gray80") +
    facet_wrap(~name, scales = "free") +
    theme_minimal()
plot_hist

# Boxplot for outliers
plot_box <- df_long %>% 
  ggplot(aes(x = name, y = value)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 1) +
  facet_wrap(~name, scales = "free") +
  theme_minimal()
plot_box


# scatterplot - not used, nothing is correlated

for (i in 1:nrow(cor_df)) {
  x_var <- cor_df$Var1[i]
  y_var <- cor_df$Var2[i]
  
  p <- ggplot(df, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.6) +
    labs(title = paste("Scatterplot of", x_var, "vs", y_var, "\nCorrelation:", round(cor_df$Freq[i], 2))) +
    theme_minimal()
  
  print(p)
}


# note; the expanded dataset included financial indicators, which were
# correlated as follows: 
#         Var1           Var2      Freq
#1      euribor3m   emp.var.rate 0.9722447
#2   emp.var.rate      euribor3m 0.9722447
#3    nr.employed      euribor3m 0.9451544
#4      euribor3m    nr.employed 0.9451544
#5    nr.employed   emp.var.rate 0.9069701
#6   emp.var.rate    nr.employed 0.9069701
#7 cons.price.idx   emp.var.rate 0.7753342
#8   emp.var.rate cons.price.idx 0.7753342



# distributions for categorical columns, removed month because not meaningful
df_char <- select(df_final, -Month) %>%  
  select(where(is.character))
df_char_long <-df_char %>% 
  pivot_longer(everything()) %>%  
  count(name, value)
df_char_plot <- df_char_long %>% 
  ggplot(aes(x = reorder(value, n), y = n, fill = name)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~name, scales = "free") +
#  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  coord_flip()
df_char_plot

# --------------
# scatterplots
# --------------

# numeric pairs - not used, just a mess, no correlations
pairs(df_num)

ggpair <- df_num %>%
  ggpairs(lower = list(continuous = "smooth"))

# num/char pairs
numeric_cols <- df_num %>% names()
categorical_cols <- df_char %>% %>% names()

for (cat in categorical_cols) {
  for (num in numeric_cols) {
    p <- ggplot(df, aes_string(x = cat, y = num)) +
      geom_boxplot(outlier.color = "red", outlier.shape = 1) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = paste("Boxplot of", num, "by", cat))
    
    print(p)  # Show each plot
  }
}


#char pairs


I used a *log-transformed* histogram (and removed the negative balances):
  
  ``` {r bal}
summary(df_num$Avg_Balance)

df_pos_bal <- df_num %>%
  select(Avg_Balance) %>% 
  filter(Avg_Balance > 0)

df_pos_bal %>% 
  ggplot(aes(x = log10(Avg_Balance + 1))) +  # Adding 1 avoids log(0) error
  geom_histogram(bins = 30, fill = "blue") +
  theme_minimal() +
  labs(title = "Log10 Histogram of Avg Annual Balance", x = "Log10(Balance + 1)")

```
