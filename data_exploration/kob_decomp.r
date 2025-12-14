# ==============================================================================
# Canadian Alcohol Consumption Decomposition Analysis
# Shift-Share Approach for Aggregate Time Series
# ==============================================================================

library(tidyverse)
library(fixest)
library(scales)
library(patchwork)
library(readr)

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
    beverage_type = `Type of beverage`,
    metric = `Value, volume and absolute volume`
  )

# Reshape consumption data to long format
consumption_long <- consumption %>%
  pivot_longer(
    cols = matches("\\d{4}"),
    names_to = "year",
    values_to = "value"
  ) %>%
  # Extract year as first 4 digits
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

# 2.1: Real Prices by Type
prices <- consumption_long %>%
  filter(metric %in% c("Value for total per capita sales", 
                       "Volume for total per capita sales")) %>%
  pivot_wider(
    names_from = metric,
    values_from = value
  ) %>%
  rename(
    value_per_capita = `Value for total per capita sales`,
    volume_per_capita = `Volume for total per capita sales`
  ) %>%
  mutate(
    nominal_price = value_per_capita / volume_per_capita
  )

# Merge with CPI to get real prices
cpi_alcohol <- cpi %>%
  filter(`Products and product groups` == "Alcoholic beverages") %>%
  pivot_longer(
    cols = starts_with("January"),
    names_to = "year_month",
    values_to = "cpi_alcohol"
  ) %>%
  mutate(year = as.integer(str_extract(year_month, "\\d{4}"))) %>%
  select(Geography, year, cpi_alcohol)

prices_real <- prices %>%
  left_join(cpi_alcohol, by = c("Geography", "year")) %>%
  mutate(
    real_price = nominal_price / (cpi_alcohol / 100)
  )

# 2.2: Cannabis Basket Weight (post-2018)
cannabis_weight <- basket %>%
  filter(`Products and product groups` == "Recreational cannabis") %>%
  pivot_longer(
    cols = matches("^\\d{4}$"),
    names_to = "year",
    values_to = "cannabis_basket_weight"
  ) %>%
  mutate(year = as.integer(year)) %>%
  select(Geography, year, cannabis_basket_weight)

# 2.3: Housing Burden
housing_basket <- basket %>%
  filter(`Products and product groups` == "Owned accommodation") %>%
  pivot_longer(
    cols = matches("^\\d{4}$"),
    names_to = "year",
    values_to = "housing_basket_weight"
  ) %>%
  mutate(year = as.integer(year)) %>%
  select(Geography, year, housing_basket_weight)

# 2.4: Age Structure
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

# Get total alcohol consumption (absolute volume = pure alcohol)
total_alcohol <- consumption_long %>%
  filter(
    beverage_type == "Total alcoholic beverages",
    metric == "Absolute volume for total per capita sales"
  ) %>%
  select(Geography, year, total_alcohol_per_capita = value)

# Merge all components
analysis_data <- total_alcohol %>%
  left_join(adult_pop, by = c("Geography", "year")) %>%
  left_join(age_structure, by = c("Geography", "year")) %>%
  left_join(cannabis_weight, by = c("Geography", "year")) %>%
  left_join(housing_basket, by = c("Geography", "year")) %>%
  # Get average real price across all alcohol types
  left_join(
    prices_real %>%
      filter(beverage_type == "Total alcoholic beverages") %>%
      group_by(Geography, year) %>%
      summarize(avg_real_price = mean(real_price, na.rm = TRUE), .groups = "drop"),
    by = c("Geography", "year")
  ) %>%
  # Add time trend
  mutate(
    year_centered = year - 2015,
    post_covid = year >= 2020,
    post_cannabis = year >= 2018
  )

# ==============================================================================
# PART 4: DESCRIPTIVE ANALYSIS
# ==============================================================================

# Plot 1: Consumption trends by province
p1 <- ggplot(analysis_data, aes(x = year, y = total_alcohol_per_capita, 
                                 color = Geography)) +
  geom_line(linewidth = 1) +
  geom_point() +
  labs(
    title = "Total Alcohol Consumption Per Capita (Pure Alcohol)",
    subtitle = "Litres per capita, ages 15+",
    x = "Year",
    y = "Litres per Capita"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot 2: Age structure changes
age_long <- analysis_data %>%
  select(Geography, year, starts_with("share_")) %>%
  pivot_longer(
    cols = starts_with("share_"),
    names_to = "age_group",
    values_to = "share"
  ) %>%
  mutate(age_group = str_remove(age_group, "share_"))

p2 <- ggplot(age_long, aes(x = year, y = share, fill = age_group)) +
  geom_area(alpha = 0.7) +
  facet_wrap(~Geography) +
  labs(
    title = "Population Age Structure Evolution",
    x = "Year",
    y = "Share of Population",
    fill = "Age Group"
  ) +
  scale_y_continuous(labels = percent) +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot 3: Cannabis basket weight over time
p3 <- ggplot(analysis_data %>% filter(!is.na(cannabis_basket_weight)), 
             aes(x = year, y = cannabis_basket_weight, color = Geography)) +
  geom_line(linewidth = 1) +
  geom_point() +
  geom_vline(xintercept = 2018, linetype = "dashed", alpha = 0.5) +
  labs(
    title = "Cannabis Basket Weight Post-Legalization",
    subtitle = "Vertical line shows legalization (2018)",
    x = "Year",
    y = "Basket Weight (%)"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# Combine plots
(p1 / p2) | p3

# Summary statistics
summary_stats <- analysis_data %>%
  group_by(Geography) %>%
  summarize(
    mean_consumption = mean(total_alcohol_per_capita, na.rm = TRUE),
    sd_consumption = sd(total_alcohol_per_capita, na.rm = TRUE),
    pct_change = (last(total_alcohol_per_capita) - first(total_alcohol_per_capita)) / 
                 first(total_alcohol_per_capita) * 100,
    mean_65plus_share = mean(`share_65_years_and_older`, na.rm = TRUE),
    change_65plus_share = last(`share_65_years_and_older`) - first(`share_65_years_and_older`)
  )

print(summary_stats)

# ==============================================================================
# PART 5: SHIFT-SHARE DECOMPOSITION
# ==============================================================================

# Define baseline (2015) and comparison (2023) periods
baseline_year <- 2015
comparison_year <- 2023

# Function to perform shift-share decomposition
shift_share_decomp <- function(data, province) {
  
  # Filter for province
  df <- data %>% filter(Geography == province)
  
  # Get baseline and comparison observations
  baseline <- df %>% filter(year == baseline_year)
  comparison <- df %>% filter(year == comparison_year)
  
  # Calculate actual change
  actual_change <- comparison$total_alcohol_per_capita - baseline$total_alcohol_per_capita
  
  # Component 1: Age structure effect
  # What if only age structure changed, keeping consumption rates constant?
  age_effect <- (comparison$`share_65_years_and_older` - baseline$`share_65_years_and_older`) * 
                baseline$total_alcohol_per_capita * -0.5  # Assume 65+ drink 50% less
  
  # Component 2: Cannabis substitution (post-2018 only)
  if (!is.na(comparison$cannabis_basket_weight)) {
    cannabis_effect <- -comparison$cannabis_basket_weight * 0.3  # Assume 30% substitution rate
  } else {
    cannabis_effect <- 0
  }
  
  # Component 3: Housing burden effect
  housing_effect <- (comparison$housing_basket_weight - baseline$housing_basket_weight) *
                    baseline$total_alcohol_per_capita * -0.2  # Assume 20% crowding out
  
  # Component 4: Residual (unexplained behavioral change)
  explained <- age_effect + cannabis_effect + housing_effect
  unexplained <- actual_change - explained
  
  # Return decomposition
  tibble(
    Geography = province,
    actual_change = actual_change,
    age_effect = age_effect,
    cannabis_effect = cannabis_effect,
    housing_effect = housing_effect,
    total_explained = explained,
    unexplained_behavioral = unexplained,
    pct_explained = explained / actual_change * 100
  )
}

# Run decomposition for all provinces
provinces <- unique(analysis_data$Geography)
decomposition_results <- map_df(provinces, ~shift_share_decomp(analysis_data, .x))

print(decomposition_results)

# ==============================================================================
# PART 6: REGRESSION-BASED APPROACH
# ==============================================================================

# Estimate consumption model
model <- feols(
  total_alcohol_per_capita ~ 
    `share_65_years_and_older` + 
    `share_18_to_21_year` +
    cannabis_basket_weight +
    housing_basket_weight +
    avg_real_price +
    post_covid | Geography,
  data = analysis_data
)

summary(model)

# Use coefficients for more precise decomposition
coefs <- coef(model)

# ==============================================================================
# PART 7: VISUALIZATION OF DECOMPOSITION
# ==============================================================================

# Waterfall chart
decomp_long <- decomposition_results %>%
  select(Geography, age_effect, cannabis_effect, housing_effect, unexplained_behavioral) %>%
  pivot_longer(
    cols = -Geography,
    names_to = "component",
    values_to = "effect"
  ) %>%
  mutate(
    component = factor(component, 
                      levels = c("age_effect", "cannabis_effect", 
                                "housing_effect", "unexplained_behavioral"),
                      labels = c("Age Structure", "Cannabis Substitution",
                                "Housing Burden", "Unexplained Behavioral"))
  )

ggplot(decomp_long, aes(x = component, y = effect, fill = component)) +
  geom_col() +
  facet_wrap(~Geography, scales = "free_y") +
  labs(
    title = "Decomposition of Alcohol Consumption Change (2015-2023)",
    subtitle = "Litres per capita change",
    x = "",
    y = "Change in Consumption"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# ==============================================================================
# PART 8: EXPORT RESULTS
# ==============================================================================

# Save results
write_csv(decomposition_results, "decomposition_results.csv")
write_csv(summary_stats, "summary_statistics.csv")

# Save plots
ggsave("consumption_trends.png", p1, width = 10, height = 6)
ggsave("age_structure.png", p2, width = 12, height = 8)
ggsave("cannabis_weights.png", p3, width = 10, height = 6)