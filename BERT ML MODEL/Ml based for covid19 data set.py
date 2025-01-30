import pandas as pd
import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AutoModelForSequenceClassification
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

DetectorFactory.seed = 0

def preprocess_data(data_path, sample_size):
    # Read the data from specific path
    data = pd.read_csv(data_path, low_memory=False)

    # Drop articles without Abstract
    data = data.dropna(subset=['abstract']).reset_index(drop=True)

    # Get "sample_size" random articles and include 'paper_id'
    data = data.sample(sample_size)[['abstract', 'paper_id']]
    
    return data

# Load BERT model and tokenizer
model_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=False, output_hidden_states=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def create_vector_from_text(tokenizer, model, text, MAX_LEN=510):
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=MAX_LEN)
    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_ids = results[0]
    attention_mask = [int(i > 0) for i in input_ids]
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits, encoded_layers = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, return_dict=False)
    layer_i = 12  # The last BERT layer before the classifier.
    batch_i = 0   # Only one input in the batch.
    token_i = 0   # The first token, corresponding to [CLS]
    vector = encoded_layers[layer_i][batch_i][token_i].detach().cpu().numpy()
    return vector

def create_vector_index(data):
    vectors = []
    source_data = data.abstract.values
    for text in tqdm(source_data):
        vector = create_vector_from_text(tokenizer, model, text)
        vectors.append(vector)
    data["vectors"] = vectors
    data["vectors"] = data["vectors"].apply(lambda emb: np.array(emb))
    data["vectors"] = data["vectors"].apply(lambda emb: emb.reshape(1, -1))
    return data

def translate_text(text, text_lang, target_lang='en'):
    model_name = f"Helsinki-NLP/opus-mt-{text_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    formated_text = ">>{}<< {}".format(text_lang, text)
    translation = model.generate(**tokenizer([formated_text], return_tensors="pt", padding=True))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translation][0]
    return translated_text

def process_document(text):
    text_vect = create_vector_from_text(tokenizer, model, text)
    text_vect = np.array(text_vect).reshape(1, -1)
    return text_vect

def is_plagiarism(similarity_score, plagiarism_threshold):
    return similarity_score >= plagiarism_threshold

def check_incoming_document(incoming_document):
    text_lang = detect(incoming_document)
    language_list = ['de', 'fr', 'el', 'ja', 'ru']
    if text_lang == 'en':
        return incoming_document
    elif text_lang not in language_list:
        return None
    else:
        return translate_text(incoming_document, text_lang)

def run_plagiarism_analysis(query_text, data, plagiarism_threshold=0.8):
    top_N = 3
    document_translation = check_incoming_document(query_text)
    
    if document_translation is None:
        return None, False  # Return a default value when unsupported
    
    query_vect = process_document(document_translation)
    data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, x))
    data["similarity"] = data["similarity"].apply(lambda x: x[0][0])  # Corrected this line
    similar_articles = data.sort_values(by='similarity', ascending=False)[1:top_N + 1]
    
    # Calculate the maximum similarity score and convert it to a percentage
    similarity_score = similar_articles["similarity"].max()
    plagiarism_percentage = similarity_score * 100
    is_plagiarized = is_plagiarism(similarity_score, plagiarism_threshold)
    
    return plagiarism_percentage, is_plagiarized

# Read data & preprocess it
data_path = "cord19_df.csv"
sample_size = 100  # Define your sample size
preprocessed_data = preprocess_data(data_path, sample_size)

# Create the vector index
vector_index = create_vector_index(preprocessed_data)

# Take input from the user
new_incoming_text = input("Enter the text to check for plagiarism: ")

# Run plagiarism analysis
plagiarism_percentage, is_plagiarized = run_plagiarism_analysis(new_incoming_text, vector_index, plagiarism_threshold=0.8)

if plagiarism_percentage is None:
    print("Analysis could not be performed due to unsupported language.")
else:
    if is_plagiarized:
        print(f"Plagiarism detected! Percentage of plagiarized content: {plagiarism_percentage:.2f}%")
   