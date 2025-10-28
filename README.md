#  Weather Forecasting — XGBoost vs LSTM

This project compares **gradient boosting (XGBoost)** and **LSTM neural networks** for **short-term temperature forecasting**.  
It focuses on transforming time-series meteorological data into supervised learning format using temporal windowing, cyclic feature encoding, and autoregressive prediction.

---

##  Project Overview

The notebook explores the following workflow:

1. **Data Preparation**
   - Cleaning and transforming raw weather data.
   - Aggregating hourly readings into daily and period-based averages (morning, afternoon, night).

2. **Feature Engineering**
   - Encoding temporal features (`day`, `month`, `dayofweek`) using sinusoidal and cosinusoidal functions to capture cyclic behavior:
     \[
     x_{\sin} = \sin\left(2\pi \frac{x}{P}\right), \quad
     x_{\cos} = \cos\left(2\pi \frac{x}{P}\right)
     \]
   - Adding derived features such as temperature variation (ΔT), humidity, pressure, and wind metrics.

3. **Modeling**
   - **XGBoost** trained on flattened sliding windows of 24–42 time steps.
   - **LSTM** trained on sequential input tensors for temporal dependency learning.
   - Both models predict future temperature and its variation.

4. **Evaluation**
   - Metrics: MSE, RMSE, and R².
   - Additional reconstruction of absolute temperatures from ΔT to evaluate long-term consistency.

---

##  Example: XGBoost Predictions

Below is an example of XGBoost predictions compared to true temperature data.

<p align="center">
  <img src="assets/xgboost_predictions.png" alt="XGBoost temperature prediction" width="600"/>
</p>

- **Top plot:** Real vs predicted temperature  
- **Bottom plot:** Real vs predicted temperature variation (ΔT)

XGBoost performs strongly on this dataset, outperforming LSTM due to limited data and the efficiency of gradient boosting with engineered features.

---

## ⚖️ Results Summary

| Model  | Target | RMSE | R² | Notes |
|--------|---------|------|----|-------|
| XGBoost | ΔT | ~2.2 | 0.81 | Fast, interpretable, works well with windowed features |
| LSTM | ΔT | ~3.1 | 0.70 | Better at capturing long dependencies but needs more data |

---

##  Key Insights
- Gradient boosting (XGBoost) can **rival or outperform deep learning** on structured tabular time-series data.  
- Proper **feature engineering** (sin/cos encoding, sliding windows) is often more impactful than model complexity.  
- LSTMs shine with **larger datasets** and **continuous temporal signals**, while XGBoost remains robust and lightweight.

---

##  Technologies Used
- **Python 3.10**
- **pandas**, **NumPy**, **scikit-learn**
- **XGBoost**, **PyTorch**
- **Matplotlib**, **Seaborn**

---

##  Author
**Nawfal Benhamdane**  
📧 nawfal.benhamdane@student-cs.fr  
💻 [GitHub Profile](https://github.com/nawfalbenhamdane)
