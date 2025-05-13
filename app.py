from flask import Flask, render_template, request, redirect, url_for
import cv2
from database import insert_user, insert_embedding, create_tables, get_all_embeddings
from face_recognition import extract_face_embedding
from siamese_model import get_embedding_model
import os
import numpy as np
from tensorflow.keras.models import load_model
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    # Kullanıcı listesini veritabanından çek
    import sqlite3
    conn = sqlite3.connect('face_data.db')
    c = conn.cursor()
    c.execute('SELECT name FROM users')
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return render_template('index.html', users=users)

@app.route('/add_user', methods=['POST'])
def add_user():
    name = request.form['name']
    user_id = insert_user(name)
    # Kamera ile yüz al
    cap = cv2.VideoCapture(0)
    model = get_embedding_model()
    ret, frame = cap.read()
    if ret:
        embedding = extract_face_embedding(frame, model)
        if embedding is not None:
            insert_embedding(user_id, embedding)
    cap.release()
    # Kullanıcı listesini güncellemek için index'e yönlendirirken users parametresi ile gönder
    return redirect(url_for('index'))

# Veritabanı tablolarını başlatmak için fonksiyonu doğrudan burada çağır
create_tables()

@app.route('/recognize', methods=['POST'])
def recognize():
    cap = cv2.VideoCapture(0)
    model = get_embedding_model()
    ret, frame = cap.read()
    user_name = 'Bilinmiyor'
    if ret:
        embedding = extract_face_embedding(frame, model)
        if embedding is not None:
            # Veritabanındaki embedding'lerle karşılaştır
            data = get_all_embeddings()
            min_dist = float('inf')
            min_user = None
            for user_id, db_emb in data:
                dist = np.linalg.norm(embedding - db_emb)
                if dist < min_dist:
                    min_dist = dist
                    min_user = user_id
            if min_user is not None:
                # Kullanıcı adını bul
                import sqlite3
                conn = sqlite3.connect('face_data.db')
                c = conn.cursor()
                c.execute('SELECT name FROM users WHERE id=?', (min_user,))
                row = c.fetchone()
                if row:
                    user_name = row[0]
                conn.close()
    cap.release()
    # Kullanıcı listesini de gönder
    import sqlite3
    conn = sqlite3.connect('face_data.db')
    c = conn.cursor()
    c.execute('SELECT name FROM users')
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return render_template('index.html', recognized=user_name, users=users)

@app.route('/train', methods=['POST'])
def train():
    subprocess.run(['python', 'train.py'])
    # Kullanıcı listesini de gönder
    import sqlite3
    conn = sqlite3.connect('face_data.db')
    c = conn.cursor()
    c.execute('SELECT name FROM users')
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return render_template('index.html', trained=True, users=users)

if __name__ == '__main__':
    app.run(debug=True)
