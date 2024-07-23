# S&P 500 Volatility-Models

This project develops and evaluates various volatility forecasting models for the S&P 500 index. The primary focus is on using the GARCH model to predict future volatility and perform sensitivity analysis on the model parameters.

## Data
- **Source:** Yahoo Finance
- **Time Period:** January 1, 2020 to August 1, 2024
- **Frequency:** Daily closing prices

## Methods
1. **Data Preprocessing:**
   - Calculate log returns from adjusted closing prices.
   - Calculate realized volatility using a rolling window of 21 days.

2. **Model Selection:**
   - Implement and evaluate ARCH, GARCH, GJR-GARCH, and EGARCH models.
   - Select the best model based on RMSE values.

3. **Volatility Forecasting:**
   - Fit the best GARCH model on the historical returns.
   - Forecast volatility for the next 7 days.

4. **Sensitivity Analysis:**
   - Vary the GARCH model parameters (p and q) to observe the impact on forecasted volatility.
   - Visualize the results using a heatmap.

## Results
### Model Evaluation
| Model      | RMSE  |
|------------|-------|
| ARCH       | 0.083 |
| GARCH      | 0.075 |
| GJR-GARCH  | 0.075 |
| EGARCH     | 0.076 |

### Forecasted Volatility
- The best model (GARCH) was used to forecast the next 7 days of volatility.
- The predicted volatilities were aligned with the future dates and visualized.

### Sensitivity Analysis
- The sensitivity analysis showed how the forecasted volatility changes with different values of \( p \) and \( q \) parameters in the GARCH model.
- Darker colors representing higher volatility and lighter colors representing lower volatility.
- **Stability:** Lower values of \( p \) result in more stable volatility forecasts across different \( q \) values.
- **Sensitivity:** Higher values of \( p \) increase the sensitivity of the model, leading to higher and more variable forecasted volatilities.
- **Optimal Parameters:** The combination of \( p = 1 \) and \( q = 4 \) appears to be optimal for minimizing forecasted volatility.

<img width="818" alt="Screenshot 2024-07-23 at 11 45 15 AM" src="https://github.com/user-attachments/assets/54d28c11-6b99-48f6-8974-17824fec5274">



## Conclusion
- The GARCH model provides a robust framework for forecasting volatility.
- Sensitivity analysis helps in understanding the stability and responsiveness of the model to parameter changes.
- Future work could explore other volatility models and incorporate additional financial indicators.

