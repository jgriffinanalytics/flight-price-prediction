# Predicting Airline Ticket Prices Using Statistical Learning

Built and compared three models in R to predict airline ticket prices using **300,153 flight records**. The best model, Random Forest, predicted prices within about **₹3,610 on average and explained 97% of the variation in price**. That is less than half the error of the other linear models.

## Business Problem

Airline prices change constantly based on demand, timing, and airline strategy, which makes them hard to predict for both travelers and businesses. The goal of this project was to figure out which flight variables actually drive ticket prices, and then build a model that can estimate the price of a flight from basic information like the airline, class, and days left before departure. This is the same type of modeling behind fare-tracking tools like Google Flights.

## Data

- **Source:** [Flight Price Prediction (Kaggle)](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction), domestic flights in India from 2022
- **Size:** 300,153 rows and 11 variables
- **Features:** airline, travel class, number of stops, departure/arrival time, source/destination city, flight duration, days left before departure
- **Target:** ticket price (in Indian rupees, ₹)

## Approach

1. **Exploratory analysis:** Prices were heavily skewed, with most fares cheap and a long tail of expensive ones. A log transformation was applied to make the data usable for modeling. Early plots showed travel class, airline, and number of stops as the strongest candidates for predicting price.
2. **Prep work:** Converted all categorical variables to factors and made sure the training and test sets used the same category levels, so the models would be evaluated fairly.
3. **Modeling:** Trained three models and compared them on a test set. RMSE (the average size of the prediction error) and R² were the scoring metrics.
   - Multiple Linear Regression (simple, interpretable baseline)
   - Elastic Net using `glmnet` (regularized regression)
   - Random Forest using `ranger` (200 trees)

| Model | RMSE (log) | R² (log) | RMSE (₹) |
|---|---|---|---|
| Baseline MLR | 0.324 | 0.915 | 7,777 |
| Elastic Net (α = 0.5) | 0.324 | 0.915 | 7,703 |
| **Random Forest** | **0.185** | **0.972** | **3,610** |

## Key Findings

- **Travel class matters most.** Business class fares are far higher and more spread out than economy. Class was the single strongest predictor, followed by airline, days left before departure, and flight duration.
- **Booking late gets expensive fast.** Prices spike sharply for last-minute bookings. The Random Forest model picked up on this, but the linear models did not, which explains most of the accuracy gap between the models.
- **The simple models hit a ceiling.** Linear regression and Elastic Net performed almost identically, which told me the problem was not overfitting. Flight pricing just is not linear, so a model that can capture those non-linear patterns will perform the best.

**The log transformation reveals two clear price tiers.** The two peaks below are economy and business class:

![Raw flight price distribution](images/fig1_raw_price_distribution.png)
![Log-transformed flight price distribution](images/fig2_log_price_distribution.png)

**Class and stops both separate prices clearly:**

![Log price by travel class](images/fig3_logprice_by_class.png)
![Log price by number of stops](images/fig4_logprice_by_stops.png)

## Tools

**R**: tidyverse (ggplot2, dplyr), caret, glmnet, ranger, forcats, readxl

## Repository Contents

- [`flight_price_project_script.Rmd`](flight_price_project_script.Rmd): full analysis code
- [`flight_price_characteristics_dataset.csv`](flight_price_characteristics_dataset.csv): dataset
- [`flight_price_project_final_report.pdf`](flight_price_project_final_report.pdf): complete written report
- [`images/`](images/): figures shown above

## Future Work

Try gradient boosting (XGBoost/LightGBM), add outside factors like holidays and fuel prices, and test interaction terms to improve the simpler models.

---

*Jack Griffin · B.S.B.A. Economics, Statistics Minor · University of Central Florida*
