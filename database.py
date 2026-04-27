import sqlite3
import os
from datetime import datetime

DB_NAME = 'stock_app.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Table for search history
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            search_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Table for price alerts
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            target_price REAL NOT NULL,
            condition TEXT NOT NULL, -- 'above' or 'below'
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_query(ticker):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO user_queries (ticker) VALUES (?)', (ticker.upper(),))
    conn.commit()
    conn.close()

def get_recent_queries(limit=10):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT ticker, search_time FROM user_queries ORDER BY search_time DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def add_alert(ticker, target_price, condition):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO alerts (ticker, target_price, condition) VALUES (?, ?, ?)', 
              (ticker.upper(), target_price, condition))
    conn.commit()
    conn.close()

def get_active_alerts():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT id, ticker, target_price, condition FROM alerts WHERE is_active = 1')
    rows = c.fetchall()
    conn.close()
    return rows

def deactivate_alert(alert_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('UPDATE alerts SET is_active = 0 WHERE id = ?', (alert_id,))
    conn.commit()
    conn.close()
