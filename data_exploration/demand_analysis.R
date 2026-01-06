# ===============================================
# Panel Fixed Effects Regression for Alcohol Demand
# ===============================================
# Goal: Estimate own- and cross-price elasticities of beer, wine, spirits, coolers
# over 10 years × 6 provinces, controlling for demographics and income effects.

library(plm)        # Panel data models
library(lmtest)     # Robust standard errors
library(sandwich)   # Robust covariance matrices
library(stargazer)  # Regression tables
library(tidyverse)  # Data manipulation
library(broom)      # Tidy regression output

# ===============================================
# 1. LOAD DATA
# ===============================================

cat("Loading data...\n")
df <- read.csv("data/analysis_data.csv")

# Filter to complete years (exclude 2024, include 2014)
df <- df %>% filter(year >= 2014 & year < 2024)

# Sort by geography and year
df <- df %>% arrange(Geography, year)

cat(sprintf("✓ Loaded %d observations\n", nrow(df)))
cat(sprintf("  Time period: %d-%d\n", min(df$year), max(df$year)))
cat(sprintf("  Geographies: %d\n", length(unique(df$Geography))))

# ===============================================
# 2. TRANSFORM VARIABLES
# ===============================================

cat("\nComputing log differences...\n")

# Create beverage consumption variables
beverages <- c("beer", "wines", "spirits", "ciders_coolers_and_other_refreshment_beverages")
bev_names <- c("beer", "wine", "spirits", "coolers")

# Compute logs of quantities
for (i in 1:length(beverages)) {
  bev <- beverages[i]
  name <- bev_names[i]
  col <- paste0("consumption_", bev)
  
  if (col %in% colnames(df)) {
    df[[paste0("log_q_", name)]] <- log(df[[col]])
  }
}

# Use beverage-specific CPI data (off-premise prices)
# Off-premise = purchases from stores (most consumption)

# Check which CPI columns exist
required_cpis <- c("beer_offpremise_cpi", "wine_offpremise_cpi", 
                   "spirits_offpremise_cpi", "alcohol_offpremise_cpi")
missing_cpis <- required_cpis[!required_cpis %in% colnames(df)]

if (length(missing_cpis) > 0) {
  cat("⚠ Warning: Missing CPI columns:", paste(missing_cpis, collapse=", "), "\n")
  cat("  Falling back to alcohol_cpi for missing beverages\n")
  
  # Use fallbacks
  if (!"beer_offpremise_cpi" %in% colnames(df)) df$beer_offpremise_cpi <- df$alcohol_cpi
  if (!"wine_offpremise_cpi" %in% colnames(df)) df$wine_offpremise_cpi <- df$alcohol_cpi
  if (!"spirits_offpremise_cpi" %in% colnames(df)) df$spirits_offpremise_cpi <- df$alcohol_cpi
  if (!"alcohol_offpremise_cpi" %in% colnames(df)) df$alcohol_offpremise_cpi <- df$alcohol_cpi
}

df$log_p_beer <- log(df$beer_offpremise_cpi)
df$log_p_wine <- log(df$wine_offpremise_cpi)
df$log_p_spirits <- log(df$spirits_offpremise_cpi)
df$log_p_coolers <- log(df$alcohol_offpremise_cpi)  # Use general alcohol for coolers

cat("✓ Using beverage-specific CPI data:\n")
cat("  - Beer: beer_offpremise_cpi\n")
cat("  - Wine: wine_offpremise_cpi\n")
cat("  - Spirits: spirits_offpremise_cpi\n")
cat("  - Coolers: alcohol_offpremise_cpi (proxy)\n")

# Compute log of mean income (already in data)
if ("mean_income" %in% colnames(df)) {
  df$log_income <- log(df$mean_income)
} else {
  cat("⚠ Warning: mean_income not found, skipping income variable\n")
}

# Demographics (already in levels)
# share_65plus, share_18_21

# ===============================================
# 3. USE LEVELS WITH FIXED EFFECTS (NO DIFFERENCING)
# ===============================================

cat("\nUsing levels with fixed effects (no differencing)...\n")
cat("  Rationale: FE removes trends, differencing + FE is overkill with T=10\n")
cat("  This preserves signal and avoids over-differencing\n")

# Just keep the panel data in levels - no differencing needed
df_panel <- df %>%
  arrange(Geography, year)

cat(sprintf("✓ Using %d observations in levels\n", nrow(df_panel)))

# ===============================================
# 4. PREPARE PANEL STRUCTURE
# ===============================================

cat("\nPreparing panel structure...\n")

# Convert to panel data frame
# Entity = Geography, Time = year
pdata <- pdata.frame(df_panel, index = c("Geography", "year"))

cat("✓ Panel structure created\n")
cat(sprintf("  Provinces: %d\n", pdim(pdata)$nT$n))
cat(sprintf("  Time periods: %d\n", pdim(pdata)$nT$T))

# ===============================================
# 5. SPECIFY AND RUN REGRESSION MODELS
# ===============================================

cat("\n" , rep("=", 80), "\n", sep="")
cat("RUNNING PANEL REGRESSIONS\n")
cat(rep("=", 80), "\n", sep="")

# Storage for results
results <- list()

# Function to run regression for each beverage
run_beverage_regression <- function(beverage_name, pdata) {
  cat(sprintf("\n--- %s ---\n", toupper(beverage_name)))
  
  # Dependent variable (LEVELS not differences)
  dep_var <- paste0("log_q_", beverage_name)
  
  # Check if enough variation exists
  if (all(is.na(pdata[[dep_var]]))) {
    cat(sprintf("  ⚠ No data for %s, skipping...\n", beverage_name))
    return(NULL)
  }
  
  # Independent variables
  # Own-price (LEVELS not differences)
  own_price <- paste0("log_p_", beverage_name)
  
  # KEY FIX: Keep only ONE cross-price for main substitution margin
  # Not enough degrees of freedom for full price system
  cross_price <- NULL
  if (beverage_name == "beer") {
    cross_price <- "log_p_wine"  # Beer ↔ Wine substitution
  } else if (beverage_name == "wine") {
    cross_price <- "log_p_beer"  # Wine ↔ Beer substitution
  } else if (beverage_name == "spirits") {
    cross_price <- "log_p_wine"  # Spirits ↔ Wine substitution
  } else if (beverage_name == "coolers") {
    cross_price <- "log_p_beer"  # Coolers ↔ Beer substitution
  }
  
  # Controls: real income (separate from price - no double counting)
  # Demographics
  controls <- c("log_income", "share_65plus", "share_18_21")
  
  # Full formula
  rhs <- c(own_price, cross_price, controls)
  formula_str <- paste(dep_var, "~", paste(rhs, collapse = " + "))
  
  cat(sprintf("  Formula: %s\n", formula_str))
  cat(sprintf("  Own-price: %s\n", own_price))
  if (!is.null(cross_price)) {
    cat(sprintf("  Cross-price (main substitution): %s\n", cross_price))
  }
  
  # === Fixed Effects (Within estimator) with two-way FE ===
  cat("  Running fixed effects (two-way: province + year)...\n")
  model_fe <- tryCatch({
    plm(as.formula(formula_str), data = pdata, model = "within", effect = "twoways")
  }, error = function(e) {
    cat(sprintf("    ⚠ Fixed effects failed: %s\n", e$message))
    return(NULL)
  })
  
  # Print summary
  if (!is.null(model_fe)) {
    cat("\n  === Fixed Effects Results ===\n")
    print(summary(model_fe))
    
    # Robust standard errors
    robust_se <- sqrt(diag(vcovHC(model_fe, type = "HC1")))
    
    cat("\n  Own-price elasticity:", coef(model_fe)[own_price])
    cat(" (SE:", robust_se[own_price], ")\n")
    
    if (!is.null(cross_price) && cross_price %in% names(coef(model_fe))) {
      cat("  Cross-price elasticity:", coef(model_fe)[cross_price])
      cat(" (SE:", robust_se[cross_price], ")\n")
    }
  }
  
  # Return results
  return(list(
    beverage = beverage_name,
    fixed_effects = model_fe,
    own_price_var = own_price,
    cross_price_var = cross_price
  ))
}
    
    # Robust standard errors
    robust_se <- sqrt(diag(vcovHC(model_fe, type = "HC1")))
    
    cat("\n  Own-price elasticity:", coef(model_fe)[own_price])
    cat(" (SE:", robust_se[own_price], ")\n")
  }
  
  # Return results
  return(list(
    beverage = beverage_name,
    pooled = model_pooled,
    fixed_effects = model_fe,
    first_diff = model_fd
  ))
}

# Run for each beverage
for (bev in bev_names) {
  results[[bev]] <- run_beverage_regression(bev, pdata)
}

# ===============================================
# 6. EXTRACT AND ORGANIZE COEFFICIENTS
# ===============================================

cat("\n", rep("=", 80), "\n", sep="")
cat("EXTRACTING ELASTICITIES\n")
cat(rep("=", 80), "\n", sep="")

elasticity_table <- data.frame()

for (bev in bev_names) {
  if (!is.null(results[[bev]]) && !is.null(results[[bev]]$fixed_effects)) {
    model <- results[[bev]]$fixed_effects
    
    # Get coefficients
    coefs <- coef(model)
    
    # Robust SEs
    robust_se <- sqrt(diag(vcovHC(model, type = "HC1")))
    
    # Own-price (use actual variable name from results)
    own_price_var <- results[[bev]]$own_price_var
    if (own_price_var %in% names(coefs)) {
      own_elast <- coefs[own_price_var]
      own_se <- robust_se[own_price_var]
      own_t <- own_elast / own_se
      own_p <- 2 * pt(-abs(own_t), df = model$df.residual)
      
      # Add to table
      row <- data.frame(
        beverage = bev,
        type = "own_price",
        elasticity = own_elast,
        se = own_se,
        t_stat = own_t,
        p_value = own_p
      )
      elasticity_table <- rbind(elasticity_table, row)
      
      cat(sprintf("\n%s:\n", toupper(bev)))
      cat(sprintf("  Own-price elasticity: %.3f (SE: %.3f)\n", own_elast, own_se))
      cat(sprintf("  t-statistic: %.3f, p-value: %.4f\n", own_t, own_p))
    }
    
    # Cross-price (only one per beverage now)
    cross_price_var <- results[[bev]]$cross_price_var
    if (!is.null(cross_price_var) && cross_price_var %in% names(coefs)) {
      cross_elast <- coefs[cross_price_var]
      cross_se <- robust_se[cross_price_var]
      cross_t <- cross_elast / cross_se
      cross_p <- 2 * pt(-abs(cross_t), df = model$df.residual)
      
      row <- data.frame(
        beverage = bev,
        type = paste0("cross_", gsub("log_p_", "", cross_price_var)),
        elasticity = cross_elast,
        se = cross_se,
        t_stat = cross_t,
        p_value = cross_p
      )
      elasticity_table <- rbind(elasticity_table, row)
      
      cat(sprintf("  Cross-price (%s): %.3f (SE: %.3f)\n", 
                  gsub("log_p_", "", cross_price_var), cross_elast, cross_se))
    }
    
    # Income elasticity
    if ("log_income" %in% names(coefs)) {
      income_elast <- coefs["log_income"]
      income_se <- robust_se["log_income"]
      income_t <- income_elast / income_se
      income_p <- 2 * pt(-abs(income_t), df = model$df.residual)
      
      cat(sprintf("  Income elasticity: %.3f (SE: %.3f)\n", income_elast, income_se))
      
      row <- data.frame(
        beverage = bev,
        type = "income",
        elasticity = income_elast,
        se = income_se,
        t_stat = income_t,
        p_value = income_p
      )
      elasticity_table <- rbind(elasticity_table, row)
    }
  }
}

# ===============================================
# 7. DIAGNOSTICS
# ===============================================

cat("\n", rep("=", 80), "\n", sep="")
cat("DIAGNOSTICS\n")
cat(rep("=", 80), "\n", sep="")

for (bev in bev_names) {
  if (!is.null(results[[bev]]) && !is.null(results[[bev]]$fixed_effects)) {
    model <- results[[bev]]$fixed_effects
    
    cat(sprintf("\n--- %s ---\n", toupper(bev)))
    cat(sprintf("  R-squared (within): %.4f\n", summary(model)$r.squared[1]))
    cat(sprintf("  Observations: %d\n", length(model$residuals)))
    
    # Test for serial correlation
    cat("  Testing for serial correlation...\n")
    pbgtest_result <- tryCatch({
      pbgtest(model)
    }, error = function(e) {
      cat(sprintf("    ⚠ Serial correlation test failed: %s\n", e$message))
      return(NULL)
    })
    
    if (!is.null(pbgtest_result)) {
      cat(sprintf("    Breusch-Godfrey test p-value: %.4f\n", pbgtest_result$p.value))
    }
  }
}

# ===============================================
# 8. CREATE REGRESSION TABLES
# ===============================================

cat("\n", rep("=", 80), "\n", sep="")
cat("CREATING REGRESSION TABLES\n")
cat(rep("=", 80), "\n", sep="")

# Collect fixed effects models
fe_models <- list()
for (bev in bev_names) {
  if (!is.null(results[[bev]]) && !is.null(results[[bev]]$fixed_effects)) {
    fe_models[[bev]] <- results[[bev]]$fixed_effects
  }
}

# Create stargazer table
if (length(fe_models) > 0) {
  cat("\nGenerating regression table...\n")
  
  # Text output
  stargazer(fe_models,
            type = "text",
            title = "Panel Fixed Effects Regression Results (Levels with Two-Way FE)",
            column.labels = names(fe_models),
            dep.var.labels = "ln(Quantity)",
            omit.stat = c("f", "ser"),
            digits = 3,
            out = "results/panel_regression_table.txt")
  
  # LaTeX output
  stargazer(fe_models,
            type = "latex",
            title = "Panel Fixed Effects Regression Results (Levels with Two-Way FE)",
            column.labels = names(fe_models),
            dep.var.labels = "\\ln(Quantity)",
            omit.stat = c("f", "ser"),
            digits = 3,
            out = "results/panel_regression_table.tex")
  
  cat("✓ Saved regression tables\n")
}

# ===============================================
# 9. SAVE RESULTS
# ===============================================

cat("\nSaving results...\n")

# Save elasticity table
write.csv(elasticity_table, "results/elasticity_estimates.csv", row.names = FALSE)
cat("✓ Saved elasticity_estimates.csv\n")

# Save detailed results as RDS
saveRDS(results, "results/panel_regression_results.rds")
cat("✓ Saved panel_regression_results.rds\n")

# Create summary table
summary_table <- elasticity_table %>%
  filter(type == "own_price") %>%
  select(beverage, elasticity, se, p_value) %>%
  arrange(beverage)

write.csv(summary_table, "results/own_price_elasticities.csv", row.names = FALSE)
cat("✓ Saved own_price_elasticities.csv\n")

# ===============================================
# 10. SUMMARY REPORT
# ===============================================

cat("\n", rep("=", 80), "\n", sep="")
cat("SUMMARY REPORT\n")
cat(rep("=", 80), "\n", sep="")

cat("\nOwn-Price Elasticities:\n")
for (bev in bev_names) {
  row <- summary_table %>% filter(beverage == bev)
  if (nrow(row) > 0) {
    sig <- ifelse(row$p_value < 0.001, "***",
                  ifelse(row$p_value < 0.01, "**",
                         ifelse(row$p_value < 0.05, "*",
                                ifelse(row$p_value < 0.1, ".", ""))))
    cat(sprintf("  %s: %.3f (%.3f) %s\n", 
                toupper(bev), row$elasticity, row$se, sig))
  }
}


cat("\n", rep("=", 80), "\n", sep="")
cat("ANALYSIS COMPLETE\n")
cat(rep("=", 80), "\n", sep="")