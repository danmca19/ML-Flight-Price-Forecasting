# AIRLINE TICKET PRICE PREDICTION PROJECT

## 🎯 Objective
Predict **airline ticket prices** based on travel characteristics such as airline, origin/destination city, departure and arrival time, duration, and class.  
The main goal is to identify the **best-performing regression model** and understand **which variables most influence pricing behavior**.

---

## ⚙️ 1. Data Preparation and Cleaning

| Step | Description |
|------|--------------|
| **Initial shape** | (300,153 rows, 12 columns) |
| **After cleaning** | (297,940 rows, 10 columns) |
| **Numeric features** | `duration`, `days_left` |
| **Categorical features** | `airline`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `class` |

The dataset retained **99% of valid records** after cleaning, providing a solid foundation for supervised modeling.

---

## 🧩 2. Models Evaluated

Three algorithms were trained using complete preprocessing pipelines (imputation, scaling, and encoding):

| Model | Description | Type |
|--------|-------------|------|
| **Lasso Regression** | Linear model with L1 regularization | Linear |
| **Random Forest Regressor** | Ensemble of independent trees averaging predictions | Ensemble |
| **Gradient Boosting Regressor** | Sequential tree boosting algorithm | Ensemble |

---

## 🧪 3. Cross-Validation Results

| Model | CV RMSE | Train RMSE | R² | Time (s) |
|--------|----------|-------------|------|-----------|
| 🥇 **Random Forest** | **4,338.28** | 4,307.63 | **0.96** | 16.9 |
| Gradient Boosting | 5,201.96 | 5,197.41 | 0.95 | 37.7 |
| Lasso | 6,756.54 | 6,755.14 | 0.91 | 22.9 |

✅ **Random Forest achieved the best overall performance** with excellent generalization (R² ≈ 0.96).

---

## 🏆 4. Best Model Overview

**Model:** Random Forest Regressor  
**Cross-validation RMSE:** 4,338.28  
**Test RMSE:** 4,365.10  
**R² (Train):** 0.964  
**R² (Test):** 0.963  

➡️ The model generalizes very well, capturing the main price behavior patterns across routes and conditions.

---

## 🔍 5. Feature Importance
<img src="https://raw.githubusercontent.com/danmca19/ML-Flight-Price-Forecasting/main/SHAP.png" alt="Gráfico de Importância SHAP" style="float: left; margin-right: 15px;"><img src="https://raw.githubusercontent.com/danmca19/ML-Flight-Price-Forecasting/main/SHAP.png" alt="Gráfico de Importância SHAP" style="float: left; margin-right: 15px;">
         
| Feature | Importance | Interpretation |
|----------|-------------|----------------|
| `class_Business` | 0.46 | Ticket class is the **strongest predictor** of price. |
| `class_Economy` | 0.40 | Reinforces the major influence of travel class. |
| `duration` | 0.12 | Longer flights tend to have higher fares. |
| `days_left` | 0.03 | Fewer days before departure → higher price. |
| `airline_Vistara` | 0.01 | Some airlines consistently charge premium fares. |
| `source_city_Delhi` | 0.01 | Departure city mildly impacts ticket cost. |

📊 **Interpretation:**  
The model aligns with real-world pricing patterns:  
- Ticket **class** and **flight duration** dominate price formation.  
- **Purchase anticipation** has moderate influence.  
- Airline and route play a smaller, secondary role.

---

## 💡 6. Recommendations and Strategic Insights

### 🔧 Model Improvement Recommendations
1. **Add new time-based features**
   - Include `day_of_week`, `month`, `is_holiday`, `season` to capture temporal effects.
2. **Create interaction features**
   - Combine variables such as `source_city + departure_time` to reveal specific pricing dynamics.
3. **Apply feature selection**
   - Reduce redundant one-hot encoded features to accelerate training and reduce overfitting risk.
4. **Try advanced algorithms**
   - Test **XGBoost**, **LightGBM**, or **CatBoost** for improved accuracy and efficiency.
5. **Use Bayesian hyperparameter optimization**
   - Replace RandomizedSearchCV with **Optuna** or **scikit-optimize (skopt)** for faster tuning.

---

### 📈 Business Insights
1. **Customer segmentation:** Identify users who tend to purchase last-minute tickets for targeted promotions.
2. **Price elasticity by class:** Small changes in “class” drive major price variations → opportunity for **up-selling** strategies.
3. **Revenue forecasting:** The model can simulate price and revenue under different demand scenarios.
4. **Dynamic pricing system:** Use predictions to adjust fares in near real-time and maintain competitiveness.
5. **Route profitability:** Analyze predicted vs. actual prices to find underpriced or overpriced routes.

---

## 🧾 7. Overall Conclusion

The **Random Forest model** achieved strong predictive performance (**R² ≈ 0.96**) with stable results on both training and test sets.  
The **flight class** and **duration** are the most significant factors influencing ticket price.



**Confidence level:** 95%  
*(Based on validated cross-validation metrics and consistent interpretation with market patterns.)*
