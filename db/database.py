import sqlite3
import os
from sqlite3 import Error
from uuid import uuid4
from datetime import datetime

class TradeDatabase:
    """
    A utility class to handle all database operations for the trading AI project.
    """

    @staticmethod
    def get_db_path(db_name: str = "trade_data.db"):
        """
        Constructs the full path to the database file, placing it in the same directory as this script.
        """
        # The directory where this script (market_database.py) is located.
        script_dir = os.path.dirname(__file__)
        return os.path.join(script_dir, db_name)

    @staticmethod
    def sql_connect(db_name: str = "trade_data.db"):
        """
        Creates a connection to the SQLite database.
        If the database does not exist, it will be created.
        Returns the connection and cursor objects.
        """
        db_path = TradeDatabase.get_db_path(db_name)
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            return conn, cursor
        except Error as e:
            print(f"Error connecting to database: {e}")
            return None, None

    @staticmethod
    def close_connection(conn):
        """
        Closes the database connection.
        """
        if conn:
            conn.close()


    @staticmethod
    def get_current_position(symbol: str) -> float:
        conn, cursor = TradeDatabase.sql_connect()
        cursor.execute("""
        SELECT quantity 
        FROM positions 
        WHERE underlying_symbol = ?
        ORDER BY rowid DESC 
        LIMIT 1
        """, (symbol,))
        row = cursor.fetchone()
        TradeDatabase.close_connection(conn)
        return float(row[0]) if row and row[0] is not None else 0.0


    @staticmethod
    def create_tables():
        """
        Creates all necessary tables in the database if they do not already exist.
        This is the main setup function for the database schema.
        """
        conn, cursor = TradeDatabase.sql_connect()
        if conn is None or cursor is None:
            print("Could not connect to the database. Tables not created.")
            return

        print("Setting up database tables...")

        # --- Table for News Articles ---
        # Stores the raw news data scraped from sources.
        news_articles_sql = """
        CREATE TABLE IF NOT EXISTS news_articles (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            published_at TEXT NOT NULL,
            url TEXT UNIQUE,
            source TEXT,
            instrument TEXT
        );
        """
        try:
            cursor.execute(news_articles_sql)
            print("- Table 'news_articles' is ready.")
        except Error as e:
            print(f"Error creating 'news_articles' table: {e}")

        # --- Table for Sentiment Analysis Results ---
        # Stores the output from the FinBERT sentiment model.
        sentiment_analysis_sql = """
        CREATE TABLE IF NOT EXISTS sentiment_analysis (
            id TEXT PRIMARY KEY,
            article_id TEXT NOT NULL,
            sentiment_label TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            market_signal TEXT NOT NULL,
            analysis_timestamp TEXT NOT NULL,
            FOREIGN KEY (article_id) REFERENCES news_articles (id)
        );
        """
        try:
            cursor.execute(sentiment_analysis_sql)
            print("- Table 'sentiment_analysis' is ready.")
        except Error as e:
            print(f"Error creating 'sentiment_analysis' table: {e}")

     
        # --- Table for 1-Minute Historical Data ---
        historical_data_1m_sql = """
        CREATE TABLE IF NOT EXISTS historical_data_1h (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            UNIQUE(symbol, timestamp)
        );
        """
        try:
            cursor.execute(historical_data_1m_sql)
            print("- Table 'historical_data_1h' is ready.")
        except Error as e:
            print(f"Error creating 'historical_data_1h' table: {e}")

        # --- Positions (from Tastytrade API) ---
        positions_sql = """
        CREATE TABLE IF NOT EXISTS positions (
            id TEXT PRIMARY KEY,
            account_number TEXT NOT NULL,
            instrument_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            underlying_symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            average_daily_market_close_price REAL,
            average_open_price REAL,
            average_yearly_market_close_price REAL,
            close_price REAL,
            cost_effect TEXT,
            is_frozen INTEGER,
            is_suppressed INTEGER,
            multiplier REAL,
            quantity_direction TEXT,
            restricted_quantity REAL,
            expires_at TEXT,
            realized_day_gain REAL,
            realized_day_gain_date TEXT,
            realized_day_gain_effect TEXT,
            realized_today REAL,
            realized_today_date TEXT,
            realized_today_effect TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """
    
        try:
            cursor.execute(positions_sql)
            print("- Table 'positions' is ready.")
        except Error as e:
            print(f"Error creating 'positions' table: {e}")

        # --- Table for LSTM Predictions ---
        #-- Stores predictions from the LSTM model alongside actual values for comparison.
        lstm_predictions_sql = """
        CREATE TABLE IF NOT EXISTS lstm_predictions (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            prediction_timestamp TEXT NOT NULL, -- When the prediction was made
            target_timestamp TEXT NOT NULL,     -- The timestamp the prediction is for
            predicted_value REAL NOT NULL,      -- The LSTM model's predicted price/value
            actual_value REAL,                  -- The actual price/value, can be filled in later
            model_version TEXT,                 -- To track which model version made the prediction
            is_historical INTEGER DEFAULT 0,    -- 0 = live forward prediction, 1 = historical backfill
            UNIQUE(symbol, target_timestamp, is_historical) -- Avoid duplicate entries
        );
        """

        try:
            cursor.execute(lstm_predictions_sql)
            print("- Table 'lstm_predictions' is ready.")
        except Error as e:
            print(f"Error creating 'lstm_predictions' table: {e}")

         # --- RL Experiences (Replay/Logging) ---
        rl_experiences_sql = """
        CREATE TABLE IF NOT EXISTS rl_experiences (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            t_timestamp TEXT NOT NULL,
            state_json TEXT NOT NULL,
            action INTEGER NOT NULL,
            reward REAL NOT NULL,
            next_state_json TEXT,
            done INTEGER NOT NULL,
            episode_id TEXT,
            info_json TEXT
        );
        """
        try:
            cursor.execute(rl_experiences_sql)
            print("- Table 'rl_experiences' is ready.")
        except Error as e:
            print(f"Error creating 'rl_experiences' table: {e}")

        # Ensure additional columns exist (safe migration for extended logging)
        try:
            cursor.execute("PRAGMA table_info('rl_experiences')")
            existing_cols = {row[1] for row in cursor.fetchall()}
            add_cols_sql = []
            if "position_before" not in existing_cols:
                add_cols_sql.append("ALTER TABLE rl_experiences ADD COLUMN position_before REAL")
            if "position_after" not in existing_cols:
                add_cols_sql.append("ALTER TABLE rl_experiences ADD COLUMN position_after REAL")
            if "executed_delta" not in existing_cols:
                add_cols_sql.append("ALTER TABLE rl_experiences ADD COLUMN executed_delta REAL")
            if "price_prev" not in existing_cols:
                add_cols_sql.append("ALTER TABLE rl_experiences ADD COLUMN price_prev REAL")
            if "price_next" not in existing_cols:
                add_cols_sql.append("ALTER TABLE rl_experiences ADD COLUMN price_next REAL")
            if "transaction_cost" not in existing_cols:
                add_cols_sql.append("ALTER TABLE rl_experiences ADD COLUMN transaction_cost REAL")
            if "risk_penalty" not in existing_cols:
                add_cols_sql.append("ALTER TABLE rl_experiences ADD COLUMN risk_penalty REAL")
            if "cash_pnl" not in existing_cols:
                add_cols_sql.append("ALTER TABLE rl_experiences ADD COLUMN cash_pnl REAL")
            if "policy_version" not in existing_cols:
                add_cols_sql.append("ALTER TABLE rl_experiences ADD COLUMN policy_version TEXT")
            if "decision_id" not in existing_cols:
                add_cols_sql.append("ALTER TABLE rl_experiences ADD COLUMN decision_id TEXT")
            for stmt in add_cols_sql:
                try:
                    cursor.execute(stmt)
                except Error:
                    pass
        except Error as e:
            print(f"Error migrating 'rl_experiences' columns: {e}")

        # --- RL Episodes summary ---
        rl_episodes_sql = """
        CREATE TABLE IF NOT EXISTS rl_episodes (
            id TEXT PRIMARY KEY,
            symbol TEXT,
            start_timestamp TEXT NOT NULL,
            end_timestamp TEXT,
            total_reward REAL,
            steps INTEGER,
            policy_version TEXT
        );
        """
        try:
            cursor.execute(rl_episodes_sql)
            print("- Table 'rl_episodes' is ready.")
        except Error as e:
            print(f"Error creating 'rl_episodes' table: {e}")

        # --- RL Decisions (Live/Sim) ---
        rl_decisions_sql = """
        CREATE TABLE IF NOT EXISTS rl_decisions (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            decision_timestamp TEXT NOT NULL,
            state_json TEXT,
            action TEXT NOT NULL,
            position_before REAL,
            position_after REAL,
            price REAL,
            pnl_change REAL,
            policy_version TEXT,
            confidence REAL,
            comment TEXT
        );
        """
        try:
            cursor.execute(rl_decisions_sql)
            print("- Table 'rl_decisions' is ready.")
        except Error as e:
            print(f"Error creating 'rl_decisions' table: {e}")

        # Helpful indexes
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lstm_predictions_symbol_ts ON lstm_predictions(symbol, target_timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist1h_symbol_ts ON historical_data_1h(symbol, timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_exp_symbol_ts ON rl_experiences(symbol, t_timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_decisions_symbol_ts ON rl_decisions(symbol, decision_timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_exp_decision_id ON rl_experiences(decision_id);")
        except Error as e:
            print(f"Error creating indexes: {e}")

            
        conn.commit()
        TradeDatabase.close_connection(conn)
        print("\nDatabase setup check complete.")

# This block allows you to run this file directly to initialize the database
if __name__ == '__main__':
    TradeDatabase.create_tables()