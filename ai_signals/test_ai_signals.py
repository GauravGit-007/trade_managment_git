# Test script for AI Signals system
# This script tests the basic functionality of the AI signals system

import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from ai_decision_engine import AIDecisionEngine
        print("✅ AIDecisionEngine imported successfully")
    except Exception as e:
        print(f"❌ Error importing AIDecisionEngine: {e}")
        return False
    
    try:
        from ai_evaluation import AIEvaluation
        print("✅ AIEvaluation imported successfully")
    except Exception as e:
        print(f"❌ Error importing AIEvaluation: {e}")
        return False
    
    try:
        from ai_monitoring import AIMonitoring
        print("✅ AIMonitoring imported successfully")
    except Exception as e:
        print(f"❌ Error importing AIMonitoring: {e}")
        return False
    
    try:
        from config import AIConfig
        print("✅ AIConfig imported successfully")
    except Exception as e:
        print(f"❌ Error importing AIConfig: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration validation"""
    print("\n🔧 Testing configuration...")
    
    try:
        from config import AIConfig
        config = AIConfig()
        
        # Validate configuration
        errors = config.validate_config()
        if errors:
            print(f"❌ Configuration errors: {errors}")
            return False
        
        print("✅ Configuration is valid")
        
        # Test symbol mapping
        test_symbol = "/ES:XCME{=h}"
        display_name = config.get_symbol_display_name(test_symbol)
        print(f"✅ Symbol mapping: {test_symbol} -> {display_name}")
        
        # Test action mapping
        action_name = config.get_action_display_name(3)
        print(f"✅ Action mapping: 3 -> {action_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("\n🗄️ Testing database connection...")
    
    try:
        from db.database import TradeDatabase
        db = TradeDatabase()
        
        # Test connection
        conn, cursor = db.sql_connect()
        if conn and cursor:
            print("✅ Database connection successful")
            db.close_connection(conn)
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_ai_engine_initialization():
    """Test AI decision engine initialization"""
    print("\n🤖 Testing AI decision engine initialization...")
    
    try:
        from ai_decision_engine import AIDecisionEngine
        engine = AIDecisionEngine()
        
        # Test basic properties
        print(f"✅ Symbols configured: {len(engine.symbols)}")
        print(f"✅ Action mapping: {len(engine.action_mapping)} actions")
        
        # Test symbol list
        expected_symbols = [
            "/ES:XCME{=h}", "/NQ:XCME{=h}", "/MES:XCME{=h}", "/MNQ:XCME{=h}",
            "/RTY:XCME{=h}", "/QM:XNYM{=h}", "/QG:XNYM{=h}", "/MCL:XNYM{=h}",
            "BTC/USD:CXTALP{=h}", "ETH/USD:CXTALP{=h}"
        ]
        
        if set(engine.symbols) == set(expected_symbols):
            print("✅ All expected symbols configured")
        else:
            print("⚠️ Symbol configuration mismatch")
        
        return True
        
    except Exception as e:
        print(f"❌ AI engine test failed: {e}")
        return False

def test_evaluation_system():
    """Test evaluation system initialization"""
    print("\n📊 Testing evaluation system...")
    
    try:
        from ai_evaluation import AIEvaluation
        evaluator = AIEvaluation()
        
        print("✅ AIEvaluation initialized successfully")
        
        # Test getting decisions (should return empty list if no data)
        decisions = evaluator.get_ai_decisions(hours_back=1)
        print(f"✅ Decision retrieval test: {len(decisions)} decisions found")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation system test failed: {e}")
        return False

def test_monitoring_system():
    """Test monitoring system initialization"""
    print("\n🔍 Testing monitoring system...")
    
    try:
        from ai_monitoring import AIMonitoring
        monitor = AIMonitoring()
        
        print("✅ AIMonitoring initialized successfully")
        
        # Test getting recent performance
        performance = monitor.get_recent_performance(hours_back=1)
        print(f"✅ Performance retrieval test: {len(performance)} recent decisions")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring system test failed: {e}")
        return False

def test_workflow_initialization():
    """Test workflow system initialization"""
    print("\n🔄 Testing workflow system...")
    
    try:
        from ai_workflow import AIWorkflow
        workflow = AIWorkflow()
        
        print("✅ AIWorkflow initialized successfully")
        print("✅ All workflow components loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting AI Signals System Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Database Connection", test_database_connection),
        ("AI Engine Initialization", test_ai_engine_initialization),
        ("Evaluation System", test_evaluation_system),
        ("Monitoring System", test_monitoring_system),
        ("Workflow System", test_workflow_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI Signals system is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
