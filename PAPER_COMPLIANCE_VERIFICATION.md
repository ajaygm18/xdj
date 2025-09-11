# Paper Compliance Verification

This document verifies that the PLSTM-TAL implementation complies with the research paper specifications.

## ✅ Data Preprocessing (Section 4.2)

- **OHLCV Data**: ✅ Uses Open, High, Low, Close prices and Volume
- **40 Technical Indicators**: ✅ Exactly 40 indicators (same as reference [17])
- **Min-Max Scaling**: ✅ Implemented in data preprocessing
- **CAE Dimensionality Reduction**: ✅ Implemented with contractive autoencoder
- **EEMD Denoising**: ✅ Closes price series denoised using EEMD with IMF subtraction

## ✅ Contractive Autoencoder (Section 4.3)

- **Loss Function**: ✅ `L_CAE(θ) = ∑L(x,g(h(X))) + λ∥J_h(X)∥_F^2`
- **Jacobian Penalty**: ✅ Frobenius norm of Jacobian matrix implemented
- **Regularization**: ✅ Applied only to training examples
- **Contractive Property**: ✅ Maps input neighborhood to smaller output neighborhood

## ✅ Market Focus

- **USA Market Only**: ✅ Switched from Indian market to USA S&P 500 (^GSPC)
- **Paper Timeframe**: ✅ 2005-01-01 to 2022-03-31
- **No Multi-Market**: ✅ Removed other markets, USA only

## ✅ Technical Indicators (Exactly 40)

1. BBANDS (Bollinger Bands - middle band)
2. WMA (Weighted Moving Average)
3. EMA (Exponential Moving Average)
4. DEMA (Double Exponential Moving Average)
5. KAMA (Kaufman Adaptive Moving Average)
6. MAMA (MESA Adaptive Moving Average)
7. MIDPRICE (Midpoint Price)
8. SAR (Parabolic SAR)
9. SMA (Simple Moving Average)
10. T3 (Triple Exponential Moving Average)
11. TEMA (Triple Exponential Moving Average)
12. TRIMA (Triangular Moving Average)
13. AD (Accumulation/Distribution)
14. ADOSC (Accumulation/Distribution Oscillator)
15. OBV (On Balance Volume)
16. MEDPRICE (Median Price)
17. TYPPRICE (Typical Price)
18. WCLPRICE (Weighted Close Price)
19. ADX (Average Directional Index)
20. ADXR (Average Directional Index Rating)
21. APO (Absolute Price Oscillator)
22. AROON (Aroon - average of up/down)
23. AROONOSC (Aroon Oscillator)
24. BOP (Balance of Power)
25. CCI (Commodity Channel Index)
26. CMO (Chande Momentum Oscillator)
27. DX (Directional Index)
28. MACD (Moving Average Convergence Divergence)
29. MFI (Money Flow Index)
30. MINUS_DI (Minus Directional Indicator)
31. MOM (Momentum)
32. PLUS_DI (Plus Directional Indicator)
33. PPO (Percentage Price Oscillator)
34. ROC (Rate of Change)
35. RSI (Relative Strength Index)
36. STOCH (Stochastic - %K)
37. STOCHRSI (Stochastic RSI)
38. ULTOSC (Ultimate Oscillator)
39. WILLR (Williams %R)
40. LOG_RETURN (Log Return)

## ✅ Algorithm Implementation

- **EEMD Algorithm**: ✅ Follows paper's 3-step process:
  1. Decompose time series with EEMD → get IMFs
  2. Measure noise level using Sample Entropy
  3. Extract filtered series: x_f(t) = x(t) - c_i(t) where c_i has max SaEn

- **CAE Algorithm**: ✅ Implements contractive loss with Jacobian regularization
- **Bayesian Optimization**: ✅ Implemented for hyperparameter tuning

## Changes Made

1. **Reduced indicators from 46 to exactly 40** by combining multi-component indicators:
   - BBANDS: Use middle band instead of upper/middle/lower
   - MACD: Use main MACD line instead of MACD/signal/histogram
   - AROON: Use average of up/down instead of separate components
   - STOCH: Use %K instead of separate %K/%D

2. **Switched to USA market (S&P 500)** from Indian market (Reliance)

3. **Updated configuration** to use ^GSPC symbol and SP500 market code

4. **Maintained paper-compliant timeframe** (2005-01-01 to 2022-03-31)

## Verification

- ✅ Exactly 40 technical indicators
- ✅ USA market focus only
- ✅ Paper-compliant CAE with Jacobian penalty
- ✅ EEMD implementation follows algorithm
- ✅ Proper timeframe and data preprocessing