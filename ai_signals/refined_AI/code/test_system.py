# Test script for Refined AI system
# Tests database connectivity and basic functionality

import os
import sys
import sqlite3
from datetime import datetime

# Add parent directories to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
root_dir = os.path.dirname(grandparent_dir)
sys.path.append(root_dir)
from db.database import TradeDatabase

def test_database_connection():
    """Test database connection and table existence"""
    print("🔍 Testing database connection...")
    
    db = TradeDatabase()
    conn, cursor = db.sql_connect()
    
    if conn is None:
        print("❌ Failed to connect to database")
        return False
    
    print("✅ Database connection successful")
    
    # Check if required tables exist
    tables_to_check = [
        'historical_data_1h',
        'news_articles', 
        'sentiment_analysis',
        'lstm_predictions',
        'Smart_AI_decisions'
    ]
    
    for table in tables_to_check:
        try:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            result = cursor.fetchone()
            if result:
                print(f"✅ Table '{table}' exists")
            else:
                print(f"❌ Table '{table}' missing")
        except Exception as e:
            print(f"❌ Error checking table '{table}': {e}")
    
    db.close_connection(conn)
    return True

def test_data_availability():
    """Test if there's data available for processing"""
    print("\n📊 Testing data availability...")
    
    db = TradeDatabase()
    symbols = [
        "/ES:XCME{=h}", "/NQ:XCME{=h}", "/MES:XCME{=h}", "/MNQ:XCME{=h}",
        "/RTY:XCME{=h}", "/QM:XNYM{=h}", "/QG:XNYM{=h}", "/MCL:XNYM{=h}"
    ]
    
    conn, cursor = db.sql_connect()
    if conn is None:
        print("❌ Database connection failed")
        return False
    
    try:
        # Check historical data
        print("\n📈 Historical Data:")
        for symbol in symbols:
            cursor.execute("SELECT COUNT(*) FROM historical_data_1h WHERE symbol = ?", (symbol,))
            count = cursor.fetchone()[0]
            print(f"  {symbol}: {count} records")
        
        # Check news data
        print("\n📰 News Data:")
        cursor.execute("SELECT COUNT(*) FROM news_articles")
        news_count = cursor.fetchone()[0]
        print(f"  Total news articles: {news_count}")
        
        cursor.execute("SELECT COUNT(*) FROM sentiment_analysis")
        sentiment_count = cursor.fetchone()[0]
        print(f"  Total sentiment analyses: {sentiment_count}")
        
        # Check LSTM predictions
        print("\n🤖 LSTM Predictions:")
        cursor.execute("SELECT COUNT(*) FROM lstm_predictions")
        lstm_count = cursor.fetchone()[0]
        print(f"  Total LSTM predictions: {lstm_count}")
        
        # Check existing AI decisions
        print("\n🎯 AI Decisions:")
        cursor.execute("SELECT COUNT(*) FROM Smart_AI_decisions")
        ai_count = cursor.fetchone()[0]
        print(f"  Total AI decisions: {ai_count}")
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False
    finally:
        db.close_connection(conn)
    
    return True

def test_ai_processor_import():
    """Test if AI processor can be imported"""
    print("\n🤖 Testing AI processor import...")
    
    try:
        from main_processor import RefinedAIProcessor
        print("✅ RefinedAIProcessor imported successfully")
        
        # Test initialization
        processor = RefinedAIProcessor()
        print("✅ RefinedAIProcessor initialized successfully")
        print(f"✅ Processing {len(processor.symbols)} symbols")
        
        return True
    except Exception as e:
        print(f"❌ Error importing RefinedAIProcessor: {e}")
        return False

def test_accuracy_checker_import():
    """Test if accuracy checker can be imported"""
    print("\n📊 Testing accuracy checker import...")
    
    try:
        from accuracy_checker import RefinedAIAccuracyChecker
        print("✅ RefinedAIAccuracyChecker imported successfully")
        
        # Test initialization
        checker = RefinedAIAccuracyChecker()
        print("✅ RefinedAIAccuracyChecker initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error importing RefinedAIAccuracyChecker: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Refined AI System Test Suite")
    print("=" * 50)
    
    tests = [
        test_database_connection,
        test_data_availability,
        test_ai_processor_import,
        test_accuracy_checker_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! System is ready to use.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main()
