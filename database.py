import sqlite3
import numpy as np

DB_PATH = 'face_data.db'

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        embedding BLOB,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()

def insert_user(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO users (name) VALUES (?)', (name,))
    user_id = c.lastrowid
    conn.commit()
    conn.close()
    return user_id

def insert_embedding(user_id, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb_blob = embedding.astype(np.float32).tobytes()
    c.execute('INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)', (user_id, emb_blob))
    conn.commit()
    conn.close()

def get_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id, embedding FROM embeddings')
    data = c.fetchall()
    conn.close()
    return [(user_id, np.frombuffer(emb, dtype=np.float32)) for user_id, emb in data]
