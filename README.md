# How to run the New Sentiment ML Bot

## Core Scripts

1. **paper_exact_finbert_all.py** - Main script that processes all headlines with FinBERT
2. **paper_exact_strategy.py** - Trading strategy implementation with SHAP analysis
3. **check_paper_implementation.py** - Diagnostic tool to verify implementation

## ESSENTIAL CSV Files (Must Include All)

### Primary Input File

4. **gdelt_real_headlines_2015_2025.csv** (85 MB) - Combined 10-year GDELT headlines
   - Contains 193,593 real news headlines from 2015-2025
   - This is the main input file for processing

### Processed Data Files (Pre-computed, saves 2-4 hours of processing)

5. **gdelt_real_headlines_sentiment_COMPLETE.csv** (56 MB) - Headlines with FinBERT scores
   - Contains all headlines with sentiment_score, prob_positive, prob_negative, prob_neutral
   - ✅ VERIFIED: Sentiment calculations are correct (P_positive - P_negative)
   - Ready to use without modifications
6. **features_EXACT_PAPER.csv** (1.9 MB) - Aggregated daily features
   - Daily sentiment features ready for trading strategy
   - Created from the sentiment scores above
   - Ready for immediate use with paper_exact_strategy.py

### Optional: Individual Year Files (for reference/debugging)

- gdelt_real_headlines_2015.csv through gdelt_real_headlines_2025.csv
- These were combined to create the main 2015_2025 file

# QUICK START GUIDE

## Important Notes

- ✅ All data files have been verified and are ready to use
- The sentiment calculations correctly implement the paper's formula (P_positive - P_negative)
- Processing 193k headlines takes ~37 minutes on GPU, 2-4 hours on CPU. These have already been pre-processed and outputed as csv files for your convenience.

## Instructions

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy tqdm scikit-learn xgboost shap torch transformers yfinance
```

### Step 2: Verify Installation

```bash
python3 check_paper_implementation.py
```

Expected output:

```
✅ Found gdelt_real_headlines_2015_2025.csv
✅ Found COMPLETE sentiment file!
✅ features_EXACT_PAPER.csv - Ready to use!
✅ Using FinBERT sentiment scores (P_positive - P_negative)
✅ ALL DATA READY!
```

### Step 3: Run Trading Strategy

```bash
python3 paper_exact_strategy.py
```

This will:

1. Load pre-processed features from `features_EXACT_PAPER.csv`
2. Fetch market data for EUR/USD, USD/JPY, and 10-year Treasury futures
3. Train Logistic Regression and XGBoost models
4. Generate trading signals and performance metrics
5. Produce SHAP interpretability analysis

Expected runtime: 5-10 minutes

### Step 4: (Optional) Reprocess Headlines with FinBERT

If you want to regenerate the sentiment scores from scratch:

```bash
python3 paper_exact_finbert_all.py
```

⚠️ **Warning**: This will take 2-4 hours on CPU, ~37 minutes on GPU

## Expected Results

The strategy should produce:

- **EUR/USD**: Target Sharpe Ratio ~5.87
- **USD/JPY**: Target Sharpe Ratio ~4.71
- **10-year Treasury**: Target Sharpe Ratio ~4.65

Current implementation achieves approximately 10-20% of these targets due to data quality and other ambiguous parameters that are not easily quantified in the paper.

## Troubleshooting

### Common Issues:

1. **"No module named 'transformers'"**

   ```bash
   pip install transformers
   ```

2. **"No module named 'xgboost'"**

   ```bash
   pip install xgboost
   ```

3. **Out of memory during FinBERT processing**

   - Reduce batch_size in paper_exact_finbert_all.py (line 23)
   - Default is 16, try 8 or 4

4. **Market data download fails**
   - Check internet connection
   - Yahoo Finance may be temporarily unavailable
   - Try again in a few minutes
