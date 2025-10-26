# AIRLINE TICKET PRICE PREDICTION PROJECT

## 🎯 Objective
Predict **airline ticket prices** based on travel characteristics such as airline, origin/destination city, departure and arrival time, duration, and class.  
The main goal is to identify the **best-performing regression model** and understand **which variables most influence pricing behavior**.

---

##  1. Data Preparation and Cleaning

| Step | Description |
|------|--------------|
| **Initial shape** | (300,153 rows, 12 columns) |
| **After cleaning** | (297,940 rows, 10 columns) |
| **Numeric features** | `duration`, `days_left` |
| **Categorical features** | `airline`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `class` |

The dataset retained **99% of valid records** after cleaning, providing a solid foundation for supervised modeling.

---

##  2. Models Evaluated

Three algorithms were trained using complete preprocessing pipelines (imputation, scaling, and encoding):

| Model | Description | Type |
|--------|-------------|------|
| **Lasso Regression** | Linear model with L1 regularization | Linear |
| **Random Forest Regressor** | Ensemble of independent trees averaging predictions | Ensemble |
| **Gradient Boosting Regressor** | Sequential tree boosting algorithm | Ensemble |

---

##  3. Cross-Validation Results

| Model | CV RMSE | Train RMSE | R² | Time (s) |
|--------|----------|-------------|------|-----------|
| 🥇 **Random Forest** | **4,338.28** | 4,307.63 | **0.96** | 16.9 |
| Gradient Boosting | 5,201.96 | 5,197.41 | 0.95 | 37.7 |
| Lasso | 6,756.54 | 6,755.14 | 0.91 | 22.9 |

✅ **Random Forest achieved the best overall performance** with excellent generalization (R² ≈ 0.96).

---

##  4. Best Model Overview

**Model:** Random Forest Regressor  
**Cross-validation RMSE:** 4,338.28  
**Test RMSE:** 4,365.10  
**R² (Train):** 0.964  
**R² (Test):** 0.963  

➡️ The model generalizes very well, capturing the main price behavior patterns across routes and conditions.


## ✈️ Business Interpretation of Model Metrics

| **Metric** | **Value** | **Interpretation for Business Teams** |
|-------------|------------|----------------------------------------|
| **RMSE** | R$ 4,365.10 | The average total prediction error is about R$4.4K — indicating strong adherence in a market with high price volatility. |
| **MAE** | R$ 2,433.30 | On average, the model misses the real ticket price by approximately R$2.4K per flight. |
| **MAPE** | 16.66% | Predictions are, on average, within ±16.6% of the actual ticket price. |
| **Median Absolute Error** | R$ 1,105.69 | Half of the predictions have an error smaller than R$1.1K. |
| **RMSE vs. Average Price** | 20.92% | The error represents around 1/5 of the average price — acceptable for an initial pricing prediction model. |
| **Accuracy within ±10%** | 44.98% | Nearly half of the predictions are within 10% of the true price. |

---

### 💡 Practical Summary

The model achieves **strong predictive performance** in an airline market known for **extreme volatility** — driven mainly by the **airline company** and the **class of service** (Economy vs. Business).

---

### 🔄 Price Volatility Context

- **Low-cost airlines** generally maintain ticket averages between **R$800 and R$1,500**, with moderate variability (±25%).  
  → In this range, a prediction error of R$4,000 would be **significant**, since it could double or triple the real value.  

- **Mid-tier or international carriers** operate with averages around **R$4,000–R$10,000**, and variations up to ±40%.  
  → Here, an RMSE of R$4,000 means the model might predict R$8,000 for a R$10,000 ticket — an **acceptable 20% deviation** for dynamic pricing insights.  

- **Premium airlines with high-end offerings**, especially **Vistara** and **Air India**, can reach **ticket prices above R$100,000** for luxury or long-haul business class.  
  → In such extreme cases, a R$4,000 RMSE represents **less than 5% error**, which is **remarkably precise** given the vast pricing range and the influence of cabin type, route exclusivity, and booking window.



---

## 🔍 5. Feature Importance
![Gráfico de Importância SHAP](https://raw.githubusercontent.com/danmca19/ML-Flight-Price-Forecasting/main/SHAP.png)
         
| Feature | Importance | Interpretation |
|----------|-------------|----------------|
| `class_Business` | 0.46 | Ticket class is the **strongest predictor** of price. |
| `class_Economy` | 0.40 | Reinforces the major influence of travel class. |
| `duration` | 0.12 | Longer flights tend to have higher fares. |
| `days_left` | 0.03 | Fewer days before departure → higher price. |
| `airline_Vistara` | 0.01 | Some airlines consistently charge premium fares. |
| `source_city_Delhi` | 0.01 | Departure city mildly impacts ticket cost. |

 **Interpretation:**  
The model aligns with real-world pricing patterns:  
- Ticket **class** and **flight duration** dominate price formation.  
- **Purchase anticipation** has moderate influence.  
- Airline and route play a smaller, secondary role.

---

##  6. Recommendations and Strategic Insights

### 🔧 Model Improvement Recommendations
1. **Add new time-based features**
   - Include `day_of_week`, `month`, `is_holiday`, `season` to capture temporal effects.
2. **Try advanced algorithms**
   - Test **XGBoost**, **LightGBM**, or **CatBoost** for improved accuracy and efficiency.
3. **Use Bayesian hyperparameter optimization**
   - Replace RandomizedSearchCV with **Optuna** or **scikit-optimize (skopt)** for faster tuning.

---

###  Business Insights
1. **Customer segmentation:** Identify users who tend to purchase last-minute tickets for targeted promotions.
2. **Price elasticity by class:** Small changes in “class” drive major price variations → opportunity for **up-selling** strategies.
3. **Revenue forecasting:** The model can simulate price and revenue under different demand scenarios.
4. **Dynamic pricing system:** Use predictions to adjust fares in near real-time and maintain competitiveness.
5. **Route profitability:** Analyze predicted vs. actual prices to find underpriced or overpriced routes.

---

##  7. Overall Conclusion

The **Random Forest model** achieved strong predictive performance (**R² ≈ 0.96**) with stable results on both training and test sets.  
The **flight class** and **duration** are the most significant factors influencing ticket price.


##  8. Return on Investment (ROI)

The **Return on Investment (ROI)** analysis quantifies the financial impact of deploying the predictive model.  
It evaluates how much business value the model generates compared to its **total implementation cost** (including development, infrastructure, and maintenance).

---

###  Definition

The ROI measures profitability and is calculated as:

\[
ROI = \frac{Total\_Benefit - Total\_Cost}{Total\_Cost} \times 100
\]

Where:

- **Total_Benefit** → Monetary gains or savings due to model implementation.  
- **Total_Cost** → Expenses associated with building, deploying, and maintaining the model.  

> A **positive ROI** (> 0%) means the model generates financial value.  
> A **negative ROI** (< 0%) indicates the model’s cost exceeds its benefits.

---

###  Scenario

| Component | Description | Estimated Value (USD) |
|-----------|-------------|----------------------|
| Cost Reduction | Savings from improved forecast accuracy | 85,000 |
| Operational Efficiency | Time saved from process automation | 25,000 |
| **Total Benefit** | — | 110,000 |
| Model Development Cost | Data processing, training, evaluation | 35,000 |
| Deployment & Maintenance | Cloud services and monitoring | 15,000 |
| **Total Cost** | — | 50,000 |

**ROI Calculation:**

\[
ROI = \frac{110,000 - 50,000}{50,000} \times 100 = 120\%
\]

✅ **Interpretation:**  
The model yields a **120% ROI**, meaning that for every $1 invested, the company earns $2.20 in business value.

