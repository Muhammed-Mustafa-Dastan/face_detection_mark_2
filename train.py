from siamese_model import get_siamese_model
import numpy as np
from database import get_all_embeddings
from tensorflow.keras.callbacks import ModelCheckpoint

def train_siamese():
    # Veritabanından embedding ve etiketleri al
    data = get_all_embeddings()
    X, y = [], []
    for i, (user_id1, emb1) in enumerate(data):
        for j, (user_id2, emb2) in enumerate(data):
            if i >= j: continue
            X.append([emb1, emb2])
            y.append(1 if user_id1 == user_id2 else 0)
    X = np.array(X)
    y = np.array(y)
    model = get_siamese_model((128,))
    checkpoint = ModelCheckpoint('siamese_model.h5', save_best_only=True, monitor='val_loss')
    model.fit([X[:,0], X[:,1]], y, batch_size=16, epochs=10, validation_split=0.2, callbacks=[checkpoint])

if __name__ == '__main__':
    train_siamese()
