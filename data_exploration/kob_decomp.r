# ==============================================================================
# Canadian Alcohol Consumption Decomposition Analysis - REVISED v2
# With: Log Alcohol CPI, Alcohol Basket Weight (instead of housing)
# ==============================================================================

library(tidyverse)
library(fixest)
library(scales)
library(patchwork)
library(readr)

# Custom VIF function (avoids car package dependency hell)
calculate_vif <- function(model) {
  # Get model matrix (design matrix)
  X <- model.matrix(model)[, -1]  # Remove intercept
  
  # Calculate VIF for each predictor
  vif_values <- numeric(ncol(X))
  names(vif_values) <- colnames(X)
  
  for(i in 1:ncol(X)) {
    # Regress predictor i on all other predictors
    y <- X[, i]
    x_others <- X[, -i, drop = FALSE]
    
    # Handle case of single predictor
    if(ncol(x_others) == 0) {
      vif_values[i] <- 1
    } else {
      r_squared <- tryCatch({
        lm_temp <- lm(y ~ x_others)
        summary(lm_temp)$r.squared
      }, error = function(e) 0)
      
      # VIF = 1 / (1 - R²)
      vif_values[i] <- 1 / (1 - r_squared)
    }
  }
  
  return(vif_values)
}

# ==============================================================================
# PART 1: DATA PREPARATION
# ==============================================================================

# Load data
consumption <- read_csv("data/alcohol_consuption_data.csv")
population <- read_csv("data/population_estimates.csv")
income <- read_csv("data/income_data.csv")
cpi <- read_csv("data/cpi.csv")
basket <- read_csv("data/basket_weights.csv")

# Clean column names
consumption <- consumption %>%
  rename(
    metric = `Value, volume and absolute volume`
  )

# Reshape consumption data to long format
consumption_long <- consumption %>%
  pivot_longer(
    cols = matches("\\d{4}"),
    names_to = "year",
    values_to = "value"
  ) %>%
  mutate(year = as.integer(str_extract(year, "^\\d{4}")))

# Reshape population data
population_long <- population %>%
  pivot_longer(
    cols = matches("^\\d{4}$"),
    names_to = "year",
    values_to = "population"
  ) %>%
  mutate(year = as.integer(year))

# Calculate adult population (15+)
adult_pop <- population_long %>%
  filter(age_group %in% c("15 to 64 years", "65 years and older")) %>%
  group_by(Geography, year) %>%
  summarize(adult_population = sum(population), .groups = "drop")

# ==============================================================================
# PART 2: CONSTRUCT KEY VARIABLES
# ==============================================================================

# 2.1: Extract Alcohol CPI and Basket Weight
alcohol_cpi <- cpi %>%
  filter(`Products and product groups` == "Alcoholic beverages") %>%
  pivot_longer(
    cols = starts_with("January"),
    names_to = "year_month",
    values_to = "cpi_alcohol"
  ) %>%
  mutate(year = as.integer(str_extract(year_month, "\\d{4}"))) %>%
  select(Geography, year, cpi_alcohol) %>%
  mutate(
    log_cpi_alcohol = log(cpi_alcohol)
  )

# Alcohol basket weight
alcohol_basket <- basket %>%
  filter(`Products and product groups` == "Alcoholic beverages") %>%
  pivot_longer(
    cols = matches("^\\d{4}$"),
    names_to = "year",
    values_to = "alcohol_basket_weight"
  ) %>%
  mutate(year = as.integer(year)) %>%
  select(Geography, year, alcohol_basket_weight) %>%
  mutate(
    log_alcohol_basket = log(alcohol_basket_weight)
  )


# 2.3: Housing Burden - KEEP FOR OPTIONAL ROBUSTNESS CHECKS
housing_rented <- basket %>%
  filter(`Products and product groups` == "Rented accommodation") %>%
  pivot_longer(
    cols = matches("^\\d{4}$"),
    names_to = "year",
    values_to = "housing_rented_weight"
  ) %>%
  mutate(year = as.integer(year)) %>%
  select(Geography, year, housing_rented_weight) %>%
  mutate(log_housing_rented = log(housing_rented_weight))

housing_owned <- basket %>%
  filter(`Products and product groups` == "Owned accommodation") %>%
  pivot_longer(
    cols = matches("^\\d{4}$"),
    names_to = "year",
    values_to = "housing_owned_weight"
  ) %>%
  mutate(year = as.integer(year)) %>%
  select(Geography, year, housing_owned_weight) %>%
  mutate(log_housing_owned = log(housing_owned_weight))

# 2.4: Age Structure - MORE GRANULAR
age_structure <- population_long %>%
  group_by(Geography, year) %>%
  mutate(
    total_pop = sum(population),
    age_share = population / total_pop
  ) %>%
  select(Geography, year, age_group, age_share) %>%
  pivot_wider(
    names_from = age_group,
    values_from = age_share,
    names_prefix = "share_"
  )

# Clean age group names
age_structure <- age_structure %>%
  rename_with(~str_replace_all(., " ", "_"), starts_with("share_"))

# ==============================================================================
# PART 3: CREATE ANALYSIS DATASET
# ==============================================================================

# Get consumption by beverage type
consumption_by_type <- consumption_long %>%
  filter(
    `beverage_type` %in% c("Total alcoholic beverages", "Beer", "Wines", "Spirits"),
    metric == "Absolute volume for total per capita sales"
  ) %>%
  select(Geography, year, beverage_type = `beverage_type`, consumption_per_capita = value) %>%
  pivot_wider(
    names_from = beverage_type,
    values_from = consumption_per_capita,
    names_prefix = "alcohol_"
  )

# Standardize column names (remove spaces)
consumption_by_type <- consumption_by_type %>%
  rename_with(~str_replace_all(., " ", "_"), starts_with("alcohol_"))

# Check what columns we actually have
actual_beverage_cols <- names(consumption_by_type)[grepl("^alcohol_", names(consumption_by_type))]
cat("\nActual beverage columns created:\n")
print(actual_beverage_cols)

# Merge all components
analysis_data <- consumption_by_type %>%
  left_join(adult_pop, by = c("Geography", "year")) %>%
  left_join(age_structure, by = c("Geography", "year")) %>%
  left_join(alcohol_cpi, by = c("Geography", "year")) %>%
  left_join(alcohol_basket, by = c("Geography", "year")) %>%
  left_join(housing_rented, by = c("Geography", "year")) %>%
  left_join(housing_owned, by = c("Geography", "year")) %>%
  mutate(
    year_centered = year - 2015,
    post_covid = ifelse(year >= 2020, 1, 0),
    # Create interaction terms
    log_cpi_x_65plus = log_cpi_alcohol * `share_65_years_and_older`,
    log_basket_x_65plus = log_alcohol_basket * `share_65_years_and_older`
  )

# ==============================================================================
# PART 4: MULTICOLLINEARITY DIAGNOSTICS
# ==============================================================================

cat("\n=== MULTICOLLINEARITY DIAGNOSTICS ===\n\n")

# Check correlations
cat("Correlation Matrix for Key Variables:\n")
cor_vars <- analysis_data %>%
  select(
    year_centered,
    log_cpi_alcohol,
    log_alcohol_basket,
    `share_65_years_and_older`,
    post_covid
  ) %>%
  drop_na()

cor_matrix <- cor(cor_vars)
print(round(cor_matrix, 3))

# VIF for baseline model
cat("\n\nVariance Inflation Factors (VIF) - New Specification:\n")
cat("(VIF > 10 indicates severe multicollinearity)\n\n")

vif_data <- analysis_data %>% drop_na()

# Model 1: Log CPI + Log Basket Weight
model_vif_1 <- lm(
  `alcohol_Total_alcoholic_beverages` ~ 
    `share_65_years_and_older` + 
    log_cpi_alcohol +
    log_alcohol_basket +
    year_centered,
  data = vif_data
)
cat("Model 1: Log CPI + Log Alcohol Basket\n")
print(calculate_vif(model_vif_1))

# Model 2: With interactions
model_vif_2 <- lm(
  `alcohol_Total_alcoholic_beverages` ~ 
    `share_65_years_and_older` + 
    log_cpi_alcohol +
    log_alcohol_basket +
    log_cpi_x_65plus +
    log_basket_x_65plus +
    year_centered,
  data = vif_data
)
cat("\n\nModel 2: With age × price/basket interactions\n")
print(calculate_vif(model_vif_2))

# Model 3: Compare with housing (for robustness)
model_vif_3 <- lm(
  `alcohol_Total_alcoholic_beverages` ~ 
    `share_65_years_and_older` + 
    log_cpi_alcohol +
    log_alcohol_basket +
    log_housing_owned +
    log_housing_rented +
    year_centered,
  data = vif_data
)
cat("\n\nModel 3: With housing controls (robustness)\n")
print(calculate_vif(model_vif_3))

# ==============================================================================
# PART 5: TEST POOLING ASSUMPTION
# ==============================================================================

cat("\n\n=== TESTING POOLING ASSUMPTION ===\n\n")

# Test if CPI and basket coefficients differ across provinces
model_pooled <- lm(
  `alcohol_Total_alcoholic_beverages` ~ 
    `share_65_years_and_older` + 
    log_cpi_alcohol +
    log_alcohol_basket +
    post_covid +
    year_centered,
  data = analysis_data
)

model_interact <- lm(
  `alcohol_Total_alcoholic_beverages` ~ 
    `share_65_years_and_older` + 
    log_cpi_alcohol * Geography +
    log_alcohol_basket * Geography +
    post_covid +
    year_centered,
  data = analysis_data
)

# F-test for interaction terms
cat("Testing if CPI/basket coefficients differ by province:\n")
pooling_test <- anova(model_pooled, model_interact)
print(pooling_test)

if(pooling_test$`Pr(>F)`[2] < 0.05) {
  cat("\n*** Pooling rejected! Provinces have different price/basket effects.\n")
  cat("    Consider province-specific models or including interactions.\n")
} else {
  cat("\n*** Pooling is acceptable. Provinces have similar price/basket effects.\n")
}

# ==============================================================================
# PART 6: DEFINE ALTERNATIVE MODEL SPECIFICATIONS
# ==============================================================================

# Model specifications to test
model_specs <- list(
  
  # Model 1: Basic - Log CPI + Log Basket, 
  basic = list(
    formula = "~ share_65_years_and_older + log_cpi_alcohol + log_alcohol_basket + year_centered",
    name = "Basic: Log CPI + Log Basket"
  ),
  
  # Model 2: With COVID dummy
  with_covid = list(
    formula = "~ share_65_years_and_older + 
               log_cpi_alcohol + log_alcohol_basket + post_covid + year_centered",
    name = "With COVID Dummy"
  ),
  
  # Model 3: No time trend
  no_trend = list(
    formula = "~ share_65_years_and_older + 
               log_cpi_alcohol + log_alcohol_basket + post_covid",
    name = "No Time Trend"
  ),
  
  # Model 4: With interactions
  with_interactions = list(
    formula = "~ share_65_years_and_older + 
               log_cpi_alcohol + log_alcohol_basket + 
               log_cpi_x_65plus + log_basket_x_65plus +
               post_covid + year_centered",
    name = "With Age Interactions"
  ),
  
  # Model 5: With housing controls (robustness)
  with_housing = list(
    formula = "~ share_65_years_and_older + 
               log_cpi_alcohol + log_alcohol_basket + 
               log_housing_owned + log_housing_rented +
               post_covid + year_centered",
    name = "With Housing Controls"
  )
)

# ==============================================================================
# PART 7: REGRESSION-BASED DECOMPOSITION WITH BOOTSTRAP SEs
# ==============================================================================

baseline_year <- 2015
comparison_year <- 2023
provinces <- unique(analysis_data$Geography)

# Updated bootstrap decomposition function
regression_decomp <- function(data, province, beverage_type = "alcohol_Total_alcoholic_beverages", 
                              model_spec = "with_covid", n_boot = 500) {
  
  # Filter for province
  df <- data %>% filter(Geography == province)
  
  # Get dependent variable
  y_var <- beverage_type
  
  # Actual change
  actual_change <- df %>%
    filter(year %in% c(baseline_year, comparison_year)) %>%
    arrange(year) %>%
    pull(!!sym(y_var)) %>%
    {.[2] - .[1]}
  
  # Select model specification
  spec <- model_specs[[model_spec]]
  
  # Bootstrap function
  boot_decomp <- function(boot_data, indices) {
    boot_sample <- boot_data[indices, ]
    
    # Build formula dynamically
    formula_str <- paste(y_var, spec$formula)
    
    # Fit model
    tryCatch({
      mod <- lm(as.formula(formula_str), data = boot_sample)
      coefs <- coef(mod)
      
      # Get changes in covariates
      base <- boot_sample %>% filter(year == baseline_year) %>% slice(1)
      comp <- boot_sample %>% filter(year == comparison_year) %>% slice(1)
      
      if(nrow(base) == 0 | nrow(comp) == 0) return(rep(NA_real_, 10))
      
      # Calculate deltas for each variable
      delta_65plus <- comp$`share_65_years_and_older` - base$`share_65_years_and_older`
      
      # Age effect
      age_effect <- ifelse("share_65_years_and_older" %in% names(coefs),
                           coefs["share_65_years_and_older"] * delta_65plus,
                           0)
      
      
      # Log CPI effect
      if("log_cpi_alcohol" %in% names(coefs)) {
        delta_log_cpi <- comp$log_cpi_alcohol - base$log_cpi_alcohol
        log_cpi_effect <- coefs["log_cpi_alcohol"] * delta_log_cpi
      } else {
        log_cpi_effect <- 0
      }
      
      # Log Basket effect
      if("log_alcohol_basket" %in% names(coefs)) {
        delta_log_basket <- comp$log_alcohol_basket - base$log_alcohol_basket
        log_basket_effect <- coefs["log_alcohol_basket"] * delta_log_basket
      } else {
        log_basket_effect <- 0
      }
      
      # Interaction effects
      if("log_cpi_x_65plus" %in% names(coefs)) {
        delta_cpi_interact <- comp$log_cpi_x_65plus - base$log_cpi_x_65plus
        cpi_interact_effect <- coefs["log_cpi_x_65plus"] * delta_cpi_interact
      } else {
        cpi_interact_effect <- 0
      }
      
      if("log_basket_x_65plus" %in% names(coefs)) {
        delta_basket_interact <- comp$log_basket_x_65plus - base$log_basket_x_65plus
        basket_interact_effect <- coefs["log_basket_x_65plus"] * delta_basket_interact
      } else {
        basket_interact_effect <- 0
      }
      
      # Housing effects (robustness models only)
      if("log_housing_owned" %in% names(coefs)) {
        delta_housing_owned <- comp$log_housing_owned - base$log_housing_owned
        housing_owned_effect <- coefs["log_housing_owned"] * delta_housing_owned
      } else {
        housing_owned_effect <- 0
      }
      
      if("log_housing_rented" %in% names(coefs)) {
        delta_housing_rented <- comp$log_housing_rented - base$log_housing_rented
        housing_rented_effect <- coefs["log_housing_rented"] * delta_housing_rented
      } else {
        housing_rented_effect <- 0
      }
      
      # COVID effect
      if("post_covid" %in% names(coefs)) {
        delta_covid <- comp$post_covid - base$post_covid
        covid_effect <- coefs["post_covid"] * delta_covid
      } else {
        covid_effect <- 0
      }
      
      # Time trend
      if("year_centered" %in% names(coefs)) {
        trend_effect <- coefs["year_centered"] * (comparison_year - baseline_year)
      } else {
        trend_effect <- 0
      }
      
      return(c(age_effect, log_cpi_effect, log_basket_effect,
               cpi_interact_effect, basket_interact_effect,
               housing_owned_effect, housing_rented_effect,
               covid_effect, trend_effect))
    }, error = function(e) {
      return(rep(NA_real_, 10))
    })
  }
  
  # Prepare data
  boot_data <- df %>% drop_na()
  
  # Check if we have enough data
  if(nrow(boot_data) < 5) {
    warning(paste("Insufficient data for", province, "-", y_var, ": only", nrow(boot_data), "observations"))
    return(tibble(
      Geography = province,
      beverage_type = y_var,
      model_spec = spec$name,
      actual_change = actual_change,
      age_effect = NA, age_se = NA,
      log_cpi_effect = NA, log_cpi_se = NA,
      log_basket_effect = NA, log_basket_se = NA,
      cpi_interact_effect = NA, cpi_interact_se = NA,
      basket_interact_effect = NA, basket_interact_se = NA,
      housing_owned_effect = NA, housing_owned_se = NA,
      housing_rented_effect = NA, housing_rented_se = NA,
      covid_effect = NA, covid_se = NA,
      trend_effect = NA, trend_se = NA,
      total_explained = NA, se_explained = NA,
      unexplained_behavioral = NA, se_unexplained = NA,
      pct_explained = NA
    ))
  }
  
  # Test that the model can be fit once before bootstrap
  test_formula <- paste(y_var, spec$formula)
  test_model <- tryCatch({
    lm(as.formula(test_formula), data = boot_data)
  }, error = function(e) {
    warning(paste("Model formula failed for", province, "-", y_var, ":", e$message))
    return(NULL)
  })
  
  if(is.null(test_model)) {
    return(tibble(
      Geography = province,
      beverage_type = y_var,
      model_spec = spec$name,
      actual_change = actual_change,
      age_effect = NA, age_se = NA,
      log_cpi_effect = NA, log_cpi_se = NA,
      log_basket_effect = NA, log_basket_se = NA,
      cpi_interact_effect = NA, cpi_interact_se = NA,
      basket_interact_effect = NA, basket_interact_se = NA,
      housing_owned_effect = NA, housing_owned_se = NA,
      housing_rented_effect = NA, housing_rented_se = NA,
      covid_effect = NA, covid_se = NA,
      trend_effect = NA, trend_se = NA,
      total_explained = NA, se_explained = NA,
      unexplained_behavioral = NA, se_unexplained = NA,
      pct_explained = NA
    ))
  }
  
  # Run bootstrap
  set.seed(42)
  boot_results <- replicate(n_boot, {
    indices <- sample(nrow(boot_data), replace = TRUE)
    boot_decomp(boot_data, indices)
  })
  
  # Check if bootstrap failed
  if(!is.matrix(boot_results)) {
    warning(paste("Bootstrap failed for", province, "-", y_var))
    return(tibble(
      Geography = province,
      beverage_type = y_var,
      model_spec = spec$name,
      actual_change = actual_change,
      age_effect = NA, age_se = NA,
      log_cpi_effect = NA, log_cpi_se = NA,
      log_basket_effect = NA, log_basket_se = NA,
      cpi_interact_effect = NA, cpi_interact_se = NA,
      basket_interact_effect = NA, basket_interact_se = NA,
      housing_owned_effect = NA, housing_owned_se = NA,
      housing_rented_effect = NA, housing_rented_se = NA,
      covid_effect = NA, covid_se = NA,
      trend_effect = NA, trend_se = NA,
      total_explained = NA, se_explained = NA,
      unexplained_behavioral = NA, se_unexplained = NA,
      pct_explained = NA
    ))
  }
  
  # Calculate means and SEs
  boot_means <- rowMeans(boot_results, na.rm = TRUE)
  boot_ses <- apply(boot_results, 1, sd, na.rm = TRUE)

  # Calculate explained vs unexplained
  # Price effects: CPI + basket + their interactions
  # After removing cannabis, indices shift:
  # [1] = age, [2] = log_cpi, [3] = log_basket, [4] = cpi_interact, 
  # [5] = basket_interact, [6] = housing_owned, [7] = housing_rented,
  # [8] = covid, [9] = trend
  price_effects <- sum(boot_means[2:5], na.rm = TRUE)
  # Total explained: Age + price effects
  total_explained <- sum(boot_means[1:5], na.rm = TRUE)
  # Unexplained: Actual - explained - housing - COVID - trend
  unexplained <- actual_change - total_explained - boot_means[6] - boot_means[7] - boot_means[8] - boot_means[9]

  # Standard errors
  se_explained <- sqrt(sum(boot_ses[1:5]^2, na.rm = TRUE))
  se_unexplained <- sqrt(se_explained^2 + boot_ses[6]^2 + boot_ses[7]^2 + boot_ses[8]^2 + boot_ses[9]^2)
  
  tibble(
    Geography = province,
    beverage_type = y_var,
    model_spec = spec$name,
    actual_change = actual_change,
    age_effect = boot_means[1],
    age_se = boot_ses[1],
    log_cpi_effect = boot_means[3],
    log_cpi_se = boot_ses[3],
    log_basket_effect = boot_means[4],
    log_basket_se = boot_ses[4],
    cpi_interact_effect = boot_means[5],
    cpi_interact_se = boot_ses[5],
    basket_interact_effect = boot_means[6],
    basket_interact_se = boot_ses[6],
    housing_owned_effect = boot_means[7],
    housing_owned_se = boot_ses[7],
    housing_rented_effect = boot_means[8],
    housing_rented_se = boot_ses[8],
    covid_effect = boot_means[9],
    covid_se = boot_ses[9],
    trend_effect = boot_means[10],
    trend_se = boot_ses[10],
    total_explained = total_explained,
    se_explained = se_explained,
    price_effects = price_effects,
    unexplained_behavioral = unexplained,
    se_unexplained = se_unexplained,
    pct_explained = (total_explained + boot_means[9] + boot_means[10]) / actual_change * 100
  )
}

# ==============================================================================
# PART 8: RUN DECOMPOSITIONS
# ==============================================================================

cat("\n\n=== RUNNING DECOMPOSITIONS ===\n\n")

# Define beverage types to analyze
available_beverages <- names(analysis_data)[grepl("^alcohol_", names(analysis_data))]

cat("\nAvailable beverage types for analysis:\n")
print(available_beverages)

# Filter to only valid beverages
valid_beverages <- available_beverages[sapply(available_beverages, function(bev) {
  sum(!is.na(analysis_data[[bev]])) > 0
})]

valid_names <- str_replace(valid_beverages, "alcohol_", "")

cat("\nBeverages with valid data:\n")
print(valid_beverages)

# Run decompositions
cat("Running bootstrap decompositions (this will take several minutes)...\n\n")

results_list <- list()

for(model_name in names(model_specs)) {
  cat("\n--- Model:", model_specs[[model_name]]$name, "---\n")
  
  for(i in seq_along(valid_beverages)) {
    bev <- valid_beverages[i]
    bev_name <- valid_names[i]
    
    cat("  Beverage:", bev_name, "\n")
    
    results <- map_df(provinces, function(prov) {
      cat("    Province:", prov, "...\n")
      regression_decomp(analysis_data, prov, beverage_type = bev, 
                       model_spec = model_name, n_boot = 500)
    })
    
    results_list[[paste(model_name, bev_name, sep = "_")]] <- results
  }
}

# Combine all results
all_results <- bind_rows(results_list)

# ==============================================================================
# PART 9: RESULTS TABLES
# ==============================================================================

cat("\n\n=== DECOMPOSITION RESULTS ===\n\n")

# Function to format results with significance stars
format_results <- function(results_df, model_name, bev_name) {
  
  cat("\n", rep("=", 100), "\n", sep = "")
  cat("Model:", model_name, " | Beverage:", bev_name, "\n")
  cat(rep("=", 100), "\n", sep = "")
  
  results_table <- results_df %>%
    filter(model_spec == model_name, beverage_type == paste0("alcohol_", bev_name)) %>%
    mutate(
      # Calculate t-statistics
      t_age = age_effect / age_se,
      t_log_cpi = log_cpi_effect / log_cpi_se,
      t_log_basket = log_basket_effect / log_basket_se,
      t_covid = covid_effect / covid_se,
      # Calculate p-values (two-tailed, df = n_years - k)
      p_age = 2 * pt(abs(t_age), df = 8, lower.tail = FALSE),
      p_log_cpi = 2 * pt(abs(t_log_cpi), df = 8, lower.tail = FALSE),
      p_log_basket = 2 * pt(abs(t_log_basket), df = 8, lower.tail = FALSE),
      p_covid = 2 * pt(abs(t_covid), df = 8, lower.tail = FALSE)
    ) %>%
    transmute(
      Geography,
      `Actual Δ` = sprintf("%.3f", actual_change),
      Age = sprintf("%.3f%s", age_effect, case_when(
        p_age < 0.001 ~ "***",
        p_age < 0.01 ~ "**",
        p_age < 0.05 ~ "*",
        p_age < 0.1 ~ ".",
        TRUE ~ ""
      )),
      `Log CPI` = sprintf("%.3f%s", log_cpi_effect, case_when(
        p_log_cpi < 0.001 ~ "***",
        p_log_cpi < 0.01 ~ "**",
        p_log_cpi < 0.05 ~ "*",
        p_log_cpi < 0.1 ~ ".",
        TRUE ~ ""
      )),
      `Log Basket` = sprintf("%.3f%s", log_basket_effect, case_when(
        p_log_basket < 0.001 ~ "***",
        p_log_basket < 0.01 ~ "**",
        p_log_basket < 0.05 ~ "*",
        p_log_basket < 0.1 ~ ".",
        TRUE ~ ""
      )),
      `Price Total` = sprintf("%.3f", price_effects),
      COVID = sprintf("%.3f%s", covid_effect, case_when(
        p_covid < 0.001 ~ "***",
        p_covid < 0.01 ~ "**",
        p_covid < 0.05 ~ "*",
        p_covid < 0.1 ~ ".",
        TRUE ~ ""
      )),
      Trend = sprintf("%.3f", trend_effect),
      Unexplained = sprintf("%.3f", unexplained_behavioral),
      `% Explained` = sprintf("%.1f%%", pct_explained)
    )
  
  print(results_table, n = Inf)
  cat("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1\n")
  cat("Price Total = Log CPI + Log Basket + interactions\n")
}

# Print results for key specifications
for(bev_name in valid_names) {
  if(paste("with_covid", bev_name, sep = "_") %in% names(results_list)) {
    format_results(all_results, "With COVID Dummy", bev_name)
  }
  if(paste("no_trend", bev_name, sep = "_") %in% names(results_list)) {
    format_results(all_results, "No Time Trend", bev_name)
  }
  if(paste("with_interactions", bev_name, sep = "_") %in% names(results_list)) {
    format_results(all_results, "With Age Interactions", bev_name)
  }
}

# ==============================================================================
# PART 10: COMPARISON ACROSS SPECIFICATIONS
# ==============================================================================

cat("\n\n=== MODEL COMPARISON: Total Alcoholic Beverages ===\n\n")

comparison_table <- all_results %>%
  filter(beverage_type == "alcohol_Total_alcoholic_beverages") %>%
  select(Geography, model_spec, log_cpi_effect, log_basket_effect, 
         price_effects, covid_effect, pct_explained) %>%
  pivot_longer(
    cols = c(log_cpi_effect, log_basket_effect, price_effects, covid_effect, pct_explained),
    names_to = "metric",
    values_to = "value"
  ) %>%
  pivot_wider(
    names_from = model_spec,
    values_from = value
  ) %>%
  arrange(Geography, metric)

print(comparison_table, n = Inf)

# ==============================================================================
# PART 11: SAVE OUTPUTS
# ==============================================================================

# Save detailed results
write_csv(all_results, "decomposition_detailed_v2.csv")

# Save summary by beverage type
summary_by_beverage <- all_results %>%
  filter(model_spec == "With COVID Dummy") %>%
  select(Geography, beverage_type, actual_change, age_effect, 
         log_cpi_effect, log_basket_effect, price_effects, covid_effect, 
         trend_effect, unexplained_behavioral, pct_explained)

write_csv(summary_by_beverage, "decomposition_by_beverage_v2.csv")

# Save model comparison
model_comparison <- all_results %>%
  filter(beverage_type == "alcohol_Total_alcoholic_beverages") %>%
  select(Geography, model_spec, log_cpi_effect, log_basket_effect, price_effects,
         covid_effect, pct_explained)

write_csv(model_comparison, "model_comparison_v2.csv")
