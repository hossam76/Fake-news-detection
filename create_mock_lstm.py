import pickle
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer

print("Creating mock LSTM model and tokenizer for testing...")

# Create a simple tokenizer
tokenizer = Tokenizer(num_words=5000)
# Fit on some fake data
fake_texts = [
    "This is fake news about politics",
    "Real news about economy",
    "Fake story about celebrities",
    "True report on science discoveries",
    "Sports news that's accurate",
    "False claims about health products",
    "Legitimate article about international relations",
    "Misleading headline about technology",
    "Accurate reporting on local events",
    "Fabricated story about famous people",
]
tokenizer.fit_on_texts(fake_texts)

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Create directories for static files if they don't exist
os.makedirs('static', exist_ok=True)
os.makedirs('analysis_data', exist_ok=True)

# Create a simple LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=100))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Save the model
model.save('lstm_model.h5')

print("Mock LSTM model saved as 'lstm_model.h5'")
print("Mock tokenizer saved as 'tokenizer.pkl'")
print("Created directories: static/, analysis_data/")
print("You can now use the LSTM model in your application!")
