# Data Validation for AI Signals
# This module helps identify and fix data quality issues

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.database import TradeDatabase

class DataValidator:
    def __init__(self):
        """Initialize the data validator"""
        self.db = TradeDatabase()
        
    def check_historical_data_quality(self, symbol):
        """Check data quality for a specific symbol"""
        print(f"🔍 Checking data quality for {symbol}...")
        
        try:
            conn, cursor = self.db.sql_connect()
            
            # Get recent data
            query = """
                SELECT open, high, low, close, volume, timestamp
                FROM historical_data_1h 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 100
            """
            cursor.execute(query, (symbol,))
            rows = cursor.fetchall()
            self.db.close_connection(conn)
            
            if not rows:
                print(f"❌ No data found for {symbol}")
                return False
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(rows, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
            
            # Check for NaN values
            nan_counts = df.isnull().sum()
            print(f"   📊 Data points: {len(df)}")
            print(f"   🔢 NaN values: {nan_counts.to_dict()}")
            
            # Check for invalid numeric values
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            invalid_data = {}
            
            for col in numeric_cols:
                try:
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    invalid_count = numeric_series.isnull().sum()
                    invalid_data[col] = invalid_count
                except Exception as e:
                    print(f"   ❌ Error processing {col}: {e}")
                    invalid_data[col] = len(df)
            
            print(f"   ⚠️ Invalid numeric values: {invalid_data}")
            
            # Check for reasonable price ranges
            try:
                df_numeric = df.copy()
                for col in numeric_cols:
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                
                # Check for negative prices
                negative_prices = (df_numeric[['open', 'high', 'low', 'close']] < 0).any().any()
                if negative_prices:
                    print(f"   ⚠️ Found negative prices!")
                
                # Check for zero prices
                zero_prices = (df_numeric[['open', 'high', 'low', 'close']] == 0).any().any()
                if zero_prices:
                    print(f"   ⚠️ Found zero prices!")
                
                # Check for extremely high prices (potential data errors)
                max_price = df_numeric[['open', 'high', 'low', 'close']].max().max()
                if max_price > 1000000:  # Arbitrary threshold
                    print(f"   ⚠️ Found extremely high prices: {max_price}")
                
            except Exception as e:
                print(f"   ❌ Error in price validation: {e}")
            
            # Overall assessment
            total_invalid = sum(invalid_data.values())
            if total_invalid == 0:
                print(f"   ✅ Data quality is good for {symbol}")
                return True
            else:
                print(f"   ⚠️ Data quality issues found for {symbol}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking data quality for {symbol}: {e}")
            return False
    
    def check_all_symbols(self):
        """Check data quality for all trading symbols"""
        symbols = [
            "/ES:XCME{=h}", "/NQ:XCME{=h}", "/MES:XCME{=h}", "/MNQ:XCME{=h}",
            "/RTY:XCME{=h}", "/QM:XNYM{=h}", "/QG:XNYM{=h}", "/MCL:XNYM{=h}",
            "BTC/USD:CXTALP{=h}", "ETH/USD:CXTALP{=h}"
        ]
        
        print("🔍 Checking data quality for all symbols...")
        print("=" * 50)
        
        results = {}
        for symbol in symbols:
            results[symbol] = self.check_historical_data_quality(symbol)
            print()  # Add spacing
        
        # Summary
        good_symbols = [s for s, is_good in results.items() if is_good]
        bad_symbols = [s for s, is_good in results.items() if not is_good]
        
        print("📊 DATA QUALITY SUMMARY")
        print("=" * 30)
        print(f"✅ Good symbols: {len(good_symbols)}")
        print(f"❌ Problematic symbols: {len(bad_symbols)}")
        
        if bad_symbols:
            print(f"\n⚠️ Symbols with data issues:")
            for symbol in bad_symbols:
                print(f"   - {symbol}")
        
        return results
    
    def clean_historical_data(self, symbol):
        """Clean historical data by removing invalid entries"""
        print(f"🧹 Cleaning data for {symbol}...")
        
        try:
            conn, cursor = self.db.sql_connect()
            
            # Delete rows with NULL or invalid values
            delete_query = """
                DELETE FROM historical_data_1h 
                WHERE symbol = ? 
                AND (
                    open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL
                    OR open = '' OR high = '' OR low = '' OR close = '' OR volume = ''
                    OR open = 'NaN' OR high = 'NaN' OR low = 'NaN' OR close = 'NaN' OR volume = 'NaN'
                )
            """
            cursor.execute(delete_query, (symbol,))
            deleted_rows = cursor.rowcount
            
            conn.commit()
            self.db.close_connection(conn)
            
            print(f"   🗑️ Deleted {deleted_rows} invalid rows for {symbol}")
            return deleted_rows
            
        except Exception as e:
            print(f"❌ Error cleaning data for {symbol}: {e}")
            return 0
    
    def clean_all_symbols(self):
        """Clean data for all symbols"""
        symbols = [
            "/ES:XCME{=h}", "/NQ:XCME{=h}", "/MES:XCME{=h}", "/MNQ:XCME{=h}",
            "/RTY:XCME{=h}", "/QM:XNYM{=h}", "/QG:XNYM{=h}", "/MCL:XNYM{=h}",
            "BTC/USD:CXTALP{=h}", "ETH/USD:CXTALP{=h}"
        ]
        
        print("🧹 Cleaning data for all symbols...")
        print("=" * 40)
        
        total_deleted = 0
        for symbol in symbols:
            deleted = self.clean_historical_data(symbol)
            total_deleted += deleted
        
        print(f"\n📊 CLEANING SUMMARY")
        print("=" * 25)
        print(f"🗑️ Total invalid rows deleted: {total_deleted}")
        
        return total_deleted

def main():
    """Main function for data validation"""
    validator = DataValidator()
    
    print("🔍 AI SIGNALS DATA VALIDATION")
    print("=" * 40)
    
    # Check data quality
    results = validator.check_all_symbols()
    
    # Ask if user wants to clean data
    bad_symbols = [s for s, is_good in results.items() if not is_good]
    if bad_symbols:
        print(f"\n⚠️ Found data quality issues in {len(bad_symbols)} symbols")
        response = input("Do you want to clean the data? (y/N): ").strip().lower()
        
        if response == 'y':
            validator.clean_all_symbols()
            print("\n🔄 Re-checking data quality after cleaning...")
            validator.check_all_symbols()
        else:
            print("Data cleaning skipped.")
    else:
        print("\n✅ All symbols have good data quality!")

if __name__ == "__main__":
    main()


