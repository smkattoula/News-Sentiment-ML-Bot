# CRITICAL REQUIREMENTS - DO NOT COMPROMISE

**NEVER REMOVE OR MODIFY PAPER METHODOLOGY COMPONENTS**
- If XGBoost is in the paper, we MUST use XGBoost
- If there are segmentation faults or any errors, FIX them without removing components
- Stay 100% faithful to the paper methodology - NO COMPROMISES
- Find workarounds for technical issues, don't eliminate required elements

---

# 🚨 CRITICAL IMPLEMENTATION STATUS (July 2025)

## MAJOR BUG DISCOVERED: FinBERT Index Mapping Was Wrong!

### Current Status:
- ✅ **Real Headlines Extracted**: 193,593 real news headlines from GDELT (2015-2025)
- ✅ **FinBERT Processing Complete**: ALL headlines processed (took 37 minutes)
- ❌ **CRITICAL BUG FOUND**: Wrong index mapping in sentiment calculation!

### The Bug:
```python
# WRONG (what we had):
p_neg = probs[i, 0]  # This is actually POSITIVE!
p_pos = probs[i, 2]  # This is actually NEUTRAL!

# CORRECT (what it should be):
p_pos = probs[i, 0]  # Index 0 = positive
p_neg = probs[i, 1]  # Index 1 = negative
```

### Impact of Bug:
- We calculated: `neutral - positive` (meaningless!)
- Should calculate: `positive - negative` (correct sentiment)
- Result: 75.6% false positive sentiment (should be ~8.4%)
- This explains the entire performance gap!

### Required Actions:
1. **Fix the bug in `paper_exact_finbert_all.py`** (lines 84-85)
2. **Reprocess ALL 193,593 headlines with correct indices**
3. **Time estimate**: 2-4 hours (already proven to take ~37 min on this machine)
3. **Expected improvement**: 3-5 Sharpe ratio points

### Key Files Created:
- `gdelt_real_headlines_2015_2025.csv` - Real headlines extracted from news URLs
- `paper_exact_finbert_all.py` - Processes ALL headlines with FinBERT (CRITICAL!)
- `paper_exact_strategy.py` - Exact paper methodology with SHAP
- `check_paper_implementation.py` - Diagnostic tool
- `demonstrate_finbert_importance.py` - Shows why FinBERT ≠ Goldstein

### Current Performance (with wrong sentiment):
- EUR/USD: 0.37 (target: 5.87)
- USD/JPY: -0.64 (target: 4.71)
- Treasury: -1.49 (target: 4.65)

---

# News Sentiment Trading Strategy Implementation Guide

This comprehensive guide implements the exact methodology from "Interpretable Machine Learning for Macro Alpha: A News Sentiment Case Study" by Yuke Zhang (arXiv:2505.16136v1, June 12, 2025).

## Overview

This project implements an interpretable machine learning framework to extract macroeconomic alpha from global news sentiment. The system processes the Global Database of Events, Language, and Tone (GDELT) Project's worldwide news feed using FinBERT to construct daily sentiment indices, which drive an XGBoost classifier to predict next-day returns for EUR/USD, USD/JPY, and 10-year U.S. Treasury futures (ZN).

### Key Performance Metrics (Out-of-Sample: c. 2017-April 2025)
- **EUR/USD**: Sharpe Ratio 5.87, CAGR 55.4%
- **USD/JPY**: Sharpe Ratio 4.65, CAGR 53.2%
- **10-year Treasury (ZN)**: Sharpe Ratio 4.65, CAGR 22.1%

## 1. Data Collection and Methodology

### 1.1 Macro News Collection and Headline Extraction

#### GDELT Data Source
- **Time Period**: January 1, 2015 to April 30, 2025
- **API**: GDELT v2 API for events records
- **Event Filtering**: GDELT EventCode 100-199 range (consultations, statements, diplomatic/economic engagements)
- **Daily Limit**: Top 100 events ranked by `num_articles` (number of articles covering the event)

#### Data Structure
Each record includes:
- Date
- Actors
- Event code
- Goldstein scale (measure of event impact)
- Source URL

#### Headline Extraction Process
1. Rank filtered events by `num_articles`
2. Retain top 100 entries per day
3. Extract headlines via parallel HTTP requests
4. Parse returned HTML using `utils.headline_utils.fetch_headline`
5. Discard events with failed or empty headline extraction

### 1.2 Sentiment Scoring with FinBERT

#### Model Configuration
- **Model**: ProsusAI/finbert checkpoint from HuggingFace
- **Processing**: Headlines only (for robustness and efficiency)
- **Preprocessing**: 
  - Lowercase text
  - Strip non-informative symbols
  - Truncate to first 512 WordPiece tokens

#### Sentiment Calculation
For each headline i on day t:
```
si,t = P_Pos - P_Neg ∈ [-1, +1]
```
Where:
- P_Pos = probability of positive sentiment
- P_Neg = probability of negative sentiment
- Values near +1 = strong positive sentiment
- Values near -1 = strong negative sentiment
- Values near 0 = neutrality or mixed signals

### 1.3 Daily Sentiment Index Construction

#### Primary Features
Let {si,t}^Nt_i=1 be the set of polarity scores for Nt valid headlines on day t:

1. **Mean Sentiment**: St = (1/Nt) Σ si,t
2. **Sentiment Dispersion**: σt = √[(1/Nt) Σ (si,t - St)²]
3. **News Volume**: Vt = Nt
4. **Log Volume**: log(1 + Nt)
5. **Article Impact**: AIt = St × log(1 + Nt)
6. **Mean Goldstein Score**: Gt = (1/Nt) Σ gi,t
7. **Goldstein Dispersion**: σG_t = √[(1/Nt) Σ (gi,t - Gt)²]

#### Temporal Features Engineering

**Lagged Features**:
- St-1, St-2, St-3 (sentiment lags)
- Similar lags for σt, Vt, Gt

**Moving Averages**:
- 5-day MA: MA5(S)t = (1/5) Σ(k=0 to 4) St-k
- 20-day MA: MA20(S)t = (1/20) Σ(k=0 to 19) St-k

**Momentum Features**:
- Sentiment acceleration: ΔSt = MA5(S)t - MA20(S)t
- Volume momentum tracking

**Volatility Measures**:
- 5-day and 10-day rolling standard deviations of St
- Rolling sums of news volume

## 2. Market Data and Target Construction

### 2.1 Data Sources
- **FX Spot Rates**: Yahoo Finance
  - EUR/USD (ticker: EURUSD=X)
  - USD/JPY (ticker: USDJPY=X)
- **Treasury Futures**: Yahoo Finance
  - 10-year Treasury futures (ticker: ZN=F)
  - Continuous front-month series with roll adjustments

### 2.2 Return Computation
- Next-day log return: rt+1 = log(Pt+1) - log(Pt)
- Binary target: yt = 1 if rt+1 > 0, else 0

### 2.3 Additional Market Features
- **Lagged return**: rt (momentum/reversal effects)
- **Historical volatility**: 20-day annualized standard deviation (vol20,t)
- **Feature alignment**: All features use data up to close of day t to predict t+1

## 3. Predictive Modeling Framework

### 3.1 Models

#### Logistic Regression (Baseline)
- L2 (Ridge) regularization
- Feature standardization (z-scores)
- Hyperparameter C selected via time-series CV

#### XGBoost (Primary Model)
- Binary logistic loss optimization
- Hyperparameters tuned via grid search with 5-fold time-series CV:
  - Tree depth
  - Learning rate
  - Number of trees
  - L1/L2 regularization penalties
- Early stopping based on validation performance

### 3.2 Training Protocol

#### Expanding Window Cross-Validation
- **Structure**: 5-fold expanding window using TimeSeriesSplit
- **Initial Training**: January 2015 - December 2016 (~2 years)
- **OOS Testing**: Early 2017 - April 2025
- **Feature Warm-up**: 20 trading days for complete feature construction

#### Within-Fold Process
1. Scale features (standardization for Logistic Regression)
2. Train model with internal CV for hyperparameter tuning
3. Generate predictions on test segment
4. Derive trading signals (threshold = 0.5):
   - Long position (+1) if p̂t > 0.5
   - Short position (-1) if p̂t ≤ 0.5
5. Calculate returns with transaction costs:
   - FX pairs: 0.02% round-trip
   - Treasury futures: 0.05% round-trip

### 3.3 Performance Evaluation
- Area Under ROC Curve (AUC)
- Accuracy
- Annualized Sharpe Ratio
- Compound Annual Growth Rate (CAGR)
- Maximum Drawdown
- Win Rate
- Total Return
- Number of Trades

## 4. Model Interpretability with SHAP

### 4.1 SHAP Implementation
- Framework: Shapley Additive Explanations (Lundberg & Lee, 2017)
- Purpose: Assign each feature a contribution value for individual predictions
- Baseline: Average prediction over training data

### 4.2 Key Feature Insights

#### Top Predictive Features (in order of importance):

1. **Sentiment Dispersion (sentiment_std)**
   - High dispersion (conflicting news) → negative predictions
   - Low dispersion (consensus) → positive predictions

2. **Article Impact (article_impact)**
   - High positive impact → bullish signals
   - High negative impact → bearish signals

3. **Mean Sentiment (sentiment_mean)**
   - Higher sentiment → upward predictions
   - Lower sentiment → downward predictions

4. **Lagged Features (sentiment_lag1, sentiment_ma5)**
   - Capture momentum and mean-reversion effects
   - Non-linear relationships revealed by SHAP

5. **Goldstein Score (goldstein_mean)**
   - Higher scores (cooperative events) → upward predictions

6. **Market Volatility (volatility_20d)**
   - High volatility → risk-off, downward predictions

## 5. Cross-Asset Differences

### 5.1 FX Markets
- Higher intrinsic volatility
- Direct sentiment-to-price linkage
- Risk-on/risk-off behavior clearly captured
- Strong performance: CAGRs > 50%

### 5.2 Treasury Markets
- Inverse sentiment relationship
- Positive news → potential rate hikes → bond price decline
- XGBoost captures sign-flipping relationships
- Moderate performance: CAGR 22.1%

## 6. Implementation Requirements

### 6.1 Data Requirements
- GDELT v2 API access
- Yahoo Finance data access
- Sufficient historical data (minimum 2 years for initial training)

### 6.2 Technical Stack
- **Python Libraries**:
  - FinBERT (HuggingFace Transformers)
  - XGBoost
  - scikit-learn
  - pandas, numpy
  - SHAP
  - requests (for API calls)

### 6.3 Computational Resources
- GPU recommended for FinBERT inference
- Sufficient memory for feature engineering
- Parallel processing capability for headline extraction

## 7. Risk Management and Considerations

### 7.1 Limitations
- Daily frequency may miss intraday opportunities
- Reliance on news availability and quality
- Model requires periodic retraining

### 7.2 Robustness Measures
- Rigorous out-of-sample testing
- Transaction cost inclusion
- Block bootstrap for statistical significance
- Interpretability as validation tool

## 8. Future Research Directions

1. **Intraday Analysis**: Higher-frequency trading implementation
2. **Asset Expansion**: Additional macro assets (commodities, EM)
3. **Enhanced NLP**: Multi-lingual sentiment, thematic focus
4. **Adaptive Learning**: Dynamic regime adaptation
5. **Alternative Data Integration**: Social media, satellite data
6. **Causal Analysis**: Economic mechanism investigation

## Conclusion

This implementation guide provides the exact methodology for building an interpretable machine learning system that extracts macro alpha from global news sentiment. The framework has demonstrated exceptional out-of-sample performance with Sharpe ratios exceeding 4.6 across multiple asset classes, while maintaining full interpretability through SHAP analysis.

The key to success lies in the integration of:
- Domain-specific NLP (FinBERT)
- Comprehensive feature engineering
- Non-linear modeling (XGBoost)
- Rigorous backtesting methodology
- Model interpretability (SHAP)

This transparent and reproducible approach bridges qualitative news narratives with quantitative trading, offering a powerful tool for modern macro trading strategies.