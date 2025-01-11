from flask import Flask, request, jsonify
import joblib
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

# Load model and supporting files
model_path = 'Models/trained_model.pth'
encoder_path = 'Models/trained_encoder.pkl'
metadata_path = 'Models/metadata.pkl'
y_mapping_path = 'Models/y_mapping.pkl'

metadata = joblib.load(metadata_path)
input_size = metadata['input_size']
output_size = metadata['output_size']

encoder = joblib.load(encoder_path)
y_mapping_dict = joblib.load(y_mapping_path)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, output_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return self.softmax(x)

model = NeuralNetwork(input_size, output_size)
model.load_state_dict(torch.load(model_path))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])

    # Rename columns to match the trained model's expected feature names
    input_df.rename(columns={
        'department': 'Department',
        'operationalTier': 'Operational Categorization Tier 1',
        'organization': 'Organization',
        'priority': 'Priority'
    }, inplace=True)

    try:
        input_encoded = encoder.transform(input_df[['Operational Categorization Tier 1', 'Priority', 'Organization', 'Department']])
        input_summary_embeddings = get_bert_embeddings(input_df['summary'].tolist())
        input_final = np.hstack((input_encoded, input_summary_embeddings))
        input_tensor = torch.tensor(input_final, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output.data, 1)
            predicted_group = y_mapping_dict[predicted.item()]

        return jsonify({'predicted_group': predicted_group})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)


