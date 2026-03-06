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
(table2 <- xtabs(~ weight_mode + algorithm, data = data2, subset = fe == 0))
# table 3, filtering by include_filled_edges == 1
(table3 <- xtabs(~ weight_mode + algorithm, data = data2, subset = fe == 1))
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

#####################################
# Actually running the regressions here
#####################################



# 1. Formula Definitions
rhs_vars <- "(log(log(fiedler_value+1)+1) + 
              log(fiedler_value+1) + 
              log(log(trace_laplacian+1)+1) + 
              log(trace_laplacian+1) +
              fiedler_value + trace_laplacian +
              I(fiedler_value^2) + I(trace_laplacian^2) +
              I(1 / (fiedler_value + 0.001)) +
              I(1 / (trace_laplacian + 0.001)))^2"

form_time  <- as.formula(paste("log(solve_time_ms) ~", rhs_vars))
form_dec   <- as.formula(paste("log(decision_count) ~", rhs_vars))
form_ridge <- as.formula(paste("~", rhs_vars))

# 2. Ridge Helper Function
fit_ridge <- function(df, y_var) {
  x <- model.matrix(form_ridge, data = df)[, -1]
  y <- log(df[[y_var]])
  constant_cols <- apply(x, 2, var) == 0 | is.na(apply(x, 2, var))
  if (any(constant_cols)) x <- x[, !constant_cols, drop = FALSE]
  if (ncol(x) == 0) return(NULL) 
  cv.glmnet(x, y, alpha = 0)
}

# 3. Modeling Block
all_models <- data2 %>%
  group_by(algorithm, weight_mode, fe) %>%
  nest() %>%
  mutate(
    lm_time    = map(data, ~ lm(form_time, data = .x)),
    lm_dec     = map(data, ~ lm(form_dec, data = .x)),
    ridge_time = map(data, ~ fit_ridge(.x, "solve_time_ms")),
    ridge_dec  = map(data, ~ fit_ridge(.x, "decision_count"))
  )

# 4. Universal Extraction and Export Function
export_unified_csv <- function(df, model_col, is_ridge = FALSE, output_filename) {
  
  # Identify all possible 36 experimental configurations
  master_configs <- data2 %>%
    distinct(algorithm, weight_mode, fe) %>%
    mutate(config_name = paste(algorithm, weight_mode, paste0("FE", fe), sep = "_"))
  
  extracted_data <- df %>%
    mutate(results = map(!!sym(model_col), function(m) {
      if (is.null(m)) return(NULL)
      
      if (is_ridge) {
        # Extract Ridge coefficients and Deviance Ratio (R-Squared equivalent)
        c_mat <- coef(m, s = "lambda.min")
        coefs <- tibble(term = rownames(c_mat), estimate = as.vector(c_mat))
        r2    <- m$glmnet.fit$dev.ratio[which(m$lambda == m$lambda.min)]
      } else {
        # Extract OLS coefficients and R-Squared
        s     <- summary(m)
        coefs <- tibble(term = names(coef(m)), estimate = unname(coef(m)))
        r2    <- s$r.squared
      }
      
      # Combine coefficients with a blank row and R2
      bind_rows(
        coefs,
        tibble(term = "zzz_blank", estimate = NA),
        tibble(term = "zzz_R_Squared", estimate = r2)
      )
    })) %>%
    select(algorithm, weight_mode, fe, results) %>%
    unnest(results) %>%
    ungroup() %>%
    mutate(config_name = paste(algorithm, weight_mode, paste0("FE", fe), sep = "_"))
  
  # Pivot into wide format
  wide_matrix <- extracted_data %>%
    select(term, config_name, estimate) %>%
    pivot_wider(names_from = config_name, values_from = estimate)
  
  # Add missing columns for any configurations that were NULL in Ridge
  missing_cols <- setdiff(master_configs$config_name, names(wide_matrix))
  for (col in missing_cols) {
    wide_matrix[[col]] <- NA
  }
  
  # Clean up row names, handle blank row, and sort columns alphabetically
  wide_matrix <- wide_matrix %>%
    mutate(term = str_remove(term, "zzz_")) %>%
    mutate(term = ifelse(term == "blank", "", term)) %>%
    select(term, sort(master_configs$config_name))
  
  write_csv(wide_matrix, output_filename)
  message("Successfully exported: ", output_filename)
  return(wide_matrix)
}

# 5. Execution
export_unified_csv(all_models, "lm_time", FALSE, "ols_time_unified.csv")
export_unified_csv(all_models, "lm_dec", FALSE, "ols_decision_unified.csv")
export_unified_csv(all_models, "ridge_time", TRUE, "ridge_time_unified.csv")
export_unified_csv(all_models, "ridge_dec", TRUE, "ridge_decision_unified.csv")