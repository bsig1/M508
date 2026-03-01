library(tidyverse)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(broom)
library(knitr)
library(kableExtra)

data <- read_csv("sudoku_long.csv")
data$id <- seq_len(nrow(data))
data2 <- data %>% mutate(fe = include_filled_edges) %>% select(-puzzle_path, -include_filled_edges)

#########################################
# Understanding the data I have
#########################################
#table 1, looking at weight mode + algorithm
(table1 <- xtabs(~ weight_mode + algorithm, data = data2))
# table 2, filtering by include_filled_edges == 0
(table1 <- xtabs(~ weight_mode + algorithm, data = data2, subset = include_filled_edges == 0))
# table 3, filtering by include_filled_edges == 1
(table1 <- xtabs(~ weight_mode + algorithm, data = data2, subset = include_filled_edges == 1))
# so we have 1000 each, of include_filled_edges (0 or 1), 
# over six weight modes
# and over three algorithms

#########################################
# Plotting the data (to understand what values I have)
#########################################
algorithms <- unique(data$algorithm)
#when I do this with solve_time_ms, I don't see much variation
#so choosing to focus on decision_count for now
for(algo in algorithms) { 
  subset_data <- subset(data2, algorithm == algo)
  p <- ggplot(subset_data, aes(x = fiedler_value, y = decision_count)) + 
    geom_point() + 
    facet_wrap(~weight_mode, nrow = 2, ncol = 3, scales = "free") + 
    labs(title = paste("Algorithm: ", algo), 
         x = "Fiedler Value", 
         y = "Decision Value")
  print(p)
  }
plot(data2$solve_time_ms)
nrow(data2 %>% filter(solve_time_ms < 50000))
plot((data2 %>% filter(solve_time_ms < 2))$solve_time_ms)
nrow(data2 %>% filter(solve_time_ms < 2))
plot((data2 %>% filter(solve_time_ms < 1))$solve_time_ms)
#this has shown a **huge** skew in values.  
#There's about 23000 points < 2, and about 35950 points < 50000
#the max is about 185000 ... 
#so let's look at log values
plot(log(data2$solve_time_ms))
plot(log((data2 %>% filter(solve_time_ms < 2))$solve_time_ms))
#this looks like a more reasonable range.  
#We're going to use log(solve_time) as a y-variable
#still looking for how to transform decision_count
plot(data2$decision_count)
plot((data2 %>% filter(decision_count < 425))$decision_count)
nrow(data2 %>% filter(decision_count < 425)) #about 35000
# let's try a log transform here as well
plot(log((data2 %>% filter(decision_count > 0))$decision_count))
plot(log((data2 %>% filter(decision_count > 0 & decision_count < exp(6)))$decision_count))
nrow(data2 %>% filter(decision_count < exp(6))) #about 35000

#I think log-transformed y-variables are the way to go




#Now looking for how to transform x-variables
plot(data2$fiedler_value)
nrow(data2 %>% filter(fiedler_value < 2e+06)) #about 33700
plot((data2 %>% filter(fiedler_value < 2e+06))$fiedler_value)
nrow(data2 %>% filter(fiedler_value < 5e+5)) #about 33000
plot((data2 %>% filter(fiedler_value < 5e+5))$fiedler_value)
nrow(data2 %>% filter(fiedler_value < 2)) #30000
nrow(data2 %>% filter(fiedler_value > 6 & fiedler_value < 5e+5)) #3000
plot((data2 %>% filter(fiedler_value < 2))$fiedler_value) #fiedler_value zero
#there's a lot of fiedler_values concentrated around 0 and 9
#a few around 3.5,a few around exp(14.22)
plot(log((data2 %>% filter(fiedler_value > 5e+5))$fiedler_value))
#and a the rest are concentrated around exp(13.75) and exp(15.5)
plot(log((data2 %>% filter(fiedler_value > 0))$fiedler_value))
plot(log(log((data2 %>% filter(fiedler_value > 0))$fiedler_value)))
plot(log(log(data2$fiedler_value + 1)+1))
#I think I'll include fiedler_value, log(log(fiedler_value)), and log(fiedler_value)
#in the regression.  I don't think I necessarily need polynomial terms, but may try those
plot(data2$trace_laplacian) #this has enormous scale
plot(log(data2$trace_laplacian))
nrow(data2 %>% filter(trace_laplacian < exp(10))) #about 30700
#I think I will use much the same linear, log(), and log(log()) terms as fiedler_value

#################################
# Regressing by algorithm, weighting method, and filled_edges
#################################
# 1. Nest and run regressions for both Y variables
regression_results <- data2 %>%
  group_by(algorithm, weight_mode, fe) %>%
  nest() %>%
  mutate(
    # --- MODEL 1: Solve Time ---
    model_time = map(data, ~ lm(log(solve_time_ms) ~ 
                                  (log(log(fiedler_value+1)+1) + 
                                     log(fiedler_value+1) + 
                                     log(log(trace_laplacian+1)+1) + 
                                     log(trace_laplacian+1))^2 +
                                  I(fiedler_value^2) + I(trace_laplacian^2) +
                                  I(1 / (fiedler_value + 0.001)), data = .x)),
    
    # --- MODEL 2: Decision Count ---
    model_dec = map(data, ~ lm(log(decision_count) ~ 
                                 (log(log(fiedler_value+1)+1) + 
                                    log(fiedler_value+1) + 
                                    log(log(trace_laplacian+1)+1) + 
                                    log(trace_laplacian+1))^2 +
                                 I(fiedler_value^2) + I(trace_laplacian^2) +
                                 I(1 / (fiedler_value + 0.001)), data = .x))
  ) %>%                
  # Extract coefficients for both models
  mutate(coefs_time = map(model_time, coef),
         coefs_dec = map(model_dec, coef))

# 2. Extract metrics for Solve Time
model_summaries_time <- regression_results %>%                
  mutate(summary = map(model_time, summary),
         r_sq = map_dbl(summary, ~ .x$r.squared)) %>%                
  mutate(
    intercept = map_dbl(coefs_time, ~ .x["(Intercept)"]),
    slp_fie_2log = map_dbl(coefs_time, ~ .x["log(log(fiedler_value + 1) + 1)"]),
    slp_fie_log = map_dbl(coefs_time, ~ .x["log(fiedler_value + 1)"]),
    slp_trace_2log = map_dbl(coefs_time, ~ .x["log(log(trace_laplacian + 1) + 1)"]),
    slp_trace_log = map_dbl(coefs_time, ~ .x["log(trace_laplacian + 1)"]),
    slp_fie_quad = map_dbl(coefs_time, ~ .x["I(fiedler_value^2)"]),
    slp_trace_quad = map_dbl(coefs_time, ~ .x["I(trace_laplacian^2)"]),
    slp_fie_inv = map_dbl(coefs_time, ~ .x["I(1/(fiedler_value + 0.001))"])
  ) %>%
  select(algorithm, weight_mode, fe, r_sq, 
         intercept, slp_fie_2log, slp_fie_log, 
         slp_trace_2log, slp_trace_log, 
         slp_fie_quad, slp_trace_quad, slp_fie_inv)                

# 3. Extract metrics for Decision Count
model_summaries_dec <- regression_results %>%                
  mutate(summary = map(model_dec, summary),
         r_sq = map_dbl(summary, ~ .x$r.squared)) %>%                
  mutate(
    intercept = map_dbl(coefs_dec, ~ .x["(Intercept)"]),
    slp_fie_2log = map_dbl(coefs_dec, ~ .x["log(log(fiedler_value + 1) + 1)"]),
    slp_fie_log = map_dbl(coefs_dec, ~ .x["log(fiedler_value + 1)"]),
    slp_trace_2log = map_dbl(coefs_dec, ~ .x["log(log(trace_laplacian + 1) + 1)"]),
    slp_trace_log = map_dbl(coefs_dec, ~ .x["log(trace_laplacian + 1)"]),
    slp_fie_quad = map_dbl(coefs_dec, ~ .x["I(fiedler_value^2)"]),
    slp_trace_quad = map_dbl(coefs_dec, ~ .x["I(trace_laplacian^2)"]),
    slp_fie_inv = map_dbl(coefs_dec, ~ .x["I(1/(fiedler_value + 0.001))"])
  ) %>%
  select(algorithm, weight_mode, fe, r_sq, 
         intercept, slp_fie_2log, slp_fie_log, 
         slp_trace_2log, slp_trace_log, 
         slp_fie_quad, slp_trace_quad, slp_fie_inv)                

# 4. Print formatted tables
cat("--- Solve Time Model Summaries ---\n")
print(kable(model_summaries_time, digits = 3, format = "simple"))
cat("\n--- Decision Count Model Summaries ---\n")
print(kable(model_summaries_dec, digits = 3, format = "simple"))


#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################

