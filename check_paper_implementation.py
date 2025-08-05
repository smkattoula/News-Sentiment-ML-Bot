#!/usr/bin/env python3
"""
Diagnostic tool to check paper implementation status
"""

import pandas as pd
import os
import numpy as np


def check_implementation_status():
    """Check what we have vs what we need"""
    
    print("\n" + "="*80)
    print("PAPER IMPLEMENTATION STATUS CHECK")
    print("="*80)
    
    # 1. Check headline data
    print("\n1️⃣ HEADLINE DATA:")
    print("-" * 40)
    
    if os.path.exists('gdelt_real_headlines_2015_2025.csv'):
        df = pd.read_csv('gdelt_real_headlines_2015_2025.csv', nrows=5)
        total_lines = sum(1 for line in open('gdelt_real_headlines_2015_2025.csv')) - 1
        real_headlines = len(df[df['is_real_headline'] == True]) if 'is_real_headline' in df.columns else 0
        print(f"✅ Found gdelt_real_headlines_2015_2025.csv")
        print(f"   Total events: {total_lines:,}")
        print(f"   Sample has real headlines: {'is_real_headline' in df.columns}")
    else:
        print("❌ Missing gdelt_real_headlines_2015_2025.csv")
    
    # 2. Check sentiment processing
    print("\n2️⃣ SENTIMENT ANALYSIS:")
    print("-" * 40)
    
    if os.path.exists('gdelt_real_headlines_sentiment_COMPLETE.csv'):
        sent_df = pd.read_csv('gdelt_real_headlines_sentiment_COMPLETE.csv', nrows=5)
        total_lines = sum(1 for line in open('gdelt_real_headlines_sentiment_COMPLETE.csv')) - 1
        print(f"✅ Found COMPLETE sentiment file!")
        print(f"   Total processed: {total_lines:,} headlines")
        print(f"   Contains FinBERT sentiment scores")
    else:
        print("❌ Missing complete FinBERT sentiment analysis")
        print("   Need to run: python3 paper_exact_finbert_all.py")
        print("   This will take 2-4 hours")
    
    # 3. Check features
    print("\n3️⃣ FEATURE FILES:")
    print("-" * 40)
    
    if os.path.exists('features_EXACT_PAPER.csv'):
        df = pd.read_csv('features_EXACT_PAPER.csv')
        print(f"✅ features_EXACT_PAPER.csv - Ready to use!")
        print(f"   Daily aggregated features with FinBERT sentiment")
        print(f"   Records: {len(df):,}")
        if 'sentiment_mean' in df.columns:
            print(f"   Sentiment mean: {df['sentiment_mean'].mean():.3f}")
    else:
        print(f"❌ features_EXACT_PAPER.csv - Missing!")
    
    # 4. Check what we're using for sentiment
    print("\n4️⃣ SENTIMENT SOURCE ANALYSIS:")
    print("-" * 40)
    
    if os.path.exists('gdelt_real_headlines_sentiment_COMPLETE.csv'):
        df = pd.read_csv('gdelt_real_headlines_sentiment_COMPLETE.csv', nrows=100)
        
        # Verify we have FinBERT sentiment scores
        if 'sentiment_score' in df.columns:
            print("✅ Using FinBERT sentiment scores (P_positive - P_negative)")
            print(f"   Sample sentiment range: [{df['sentiment_score'].min():.3f}, {df['sentiment_score'].max():.3f}]")
            
            # Check if probabilities sum to 1
            if 'prob_positive' in df.columns and 'prob_negative' in df.columns and 'prob_neutral' in df.columns:
                prob_sum = df['prob_positive'] + df['prob_negative'] + df['prob_neutral']
                print(f"   Probability validation: sum = {prob_sum.mean():.3f} (should be 1.0)")
        else:
            print("❌ Missing sentiment_score column!")
    else:
        print("❌ Need to process headlines with FinBERT first")
    
    # 5. Performance comparison
    print("\n5️⃣ PERFORMANCE COMPARISON:")
    print("-" * 40)
    
    results_files = {
        'optimized_results.json': "Previous results",
        'real_headline_results.json': "Real headlines (Goldstein proxy)",
        'paper_exact_results.json': "EXACT paper implementation"
    }
    
    import json
    for file, desc in results_files.items():
        if os.path.exists(file):
            with open(file, 'r') as f:
                results = json.load(f)
            print(f"\n{desc}:")
            for symbol in ['EUR/USD', 'USD/JPY', '10yr_Treasury']:
                if symbol in results:
                    xgb_sharpe = results[symbol].get('xgboost', {}).get('sharpe_ratio', 
                                   results[symbol].get('xgboost', {}).get('sharpe', 0))
                    print(f"  {symbol}: {xgb_sharpe:.2f}")
    
    print(f"\nPaper target: EUR/USD 5.87, USD/JPY 4.71, Treasury 4.65")
    
    # 6. Recommendations
    print("\n6️⃣ CRITICAL NEXT STEPS:")
    print("-" * 40)
    
    if not os.path.exists('gdelt_real_headlines_sentiment_COMPLETE.csv'):
        print("1. MUST process ALL real headlines with FinBERT:")
        print("   python3 paper_exact_finbert_all.py")
        print("   ⚠️ This will take 2-4 hours but is ESSENTIAL")
        print("\n2. Then run exact paper strategy:")
        print("   python3 paper_exact_strategy.py")
    else:
        print("1. Run exact paper strategy:")
        print("   python3 paper_exact_strategy.py")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    
    if os.path.exists('gdelt_real_headlines_sentiment_COMPLETE.csv') and os.path.exists('features_EXACT_PAPER.csv'):
        print("\n✅ ALL DATA READY!")
        print("   - Real headlines: ✓")
        print("   - FinBERT sentiment scores: ✓") 
        print("   - Daily aggregated features: ✓")
        print("\n🚀 Ready to run the trading strategy!")
        print("   Execute: python3 paper_exact_strategy.py")
    else:
        missing = []
        if not os.path.exists('gdelt_real_headlines_sentiment_COMPLETE.csv'):
            missing.append("FinBERT sentiment scores")
        if not os.path.exists('features_EXACT_PAPER.csv'):
            missing.append("Daily aggregated features")
        
        print(f"\n❌ Missing: {', '.join(missing)}")
        print("\nRun: python3 paper_exact_finbert_all.py")
        print("to generate the missing data (takes 2-4 hours)")


if __name__ == "__main__":
    check_implementation_status()