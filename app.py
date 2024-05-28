
from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the saved model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        input_text = data['sentence']
        max_sequence_len = 17
        with open('tokenizer.pkl', 'rb') as f:
            mytokenizer = pickle.load(f)

        sequences = mytokenizer.texts_to_sequences([input_text])[0]

        padded_sequences = pad_sequences([sequences], maxlen=max_sequence_len-1, padding='pre')

        # Predict the scores for each sequence
        predictions = model.predict(padded_sequences)

        # Get the indices of the tokens with the top three highest scores for each sequence
        top_indices = np.argsort(-predictions, axis=1)[:, :3]

        # Convert the indices back to tokens
        top_tokens = [[mytokenizer.index_word[idx] for idx in indices] for indices in top_indices]

        # Return the prediction as JSON response
        return jsonify(top_tokens.tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

