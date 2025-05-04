import os
import re
import pickle
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data():
    # Load dataset
    data = pd.read_csv("Data/Coursera.csv")
    
    # Clean and combine text features
    text_columns = ['Course Name', 'Course Description', 'Skills']
    for col in text_columns:
        data[col] = data[col].astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', x))
        data[col] = data[col].str.lower().str.replace(r'\s+', ' ', regex=True)
    
    # Create tags with proper formatting
    data['tags'] = (
        data['Course Name'] + ' ' +
        data['Difficulty Level'] + ' ' +
        data['Course Description'] + ' ' +
        data['Skills']
    )
    
    # Stemming
    ps = PorterStemmer()
    data['tags'] = data['tags'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
    
    # --- Keep all original columns needed for app ---
    # Rename 'Course Name' to 'course_name' for compatibility
    data.rename(columns={'Course Name': 'course_name'}, inplace=True)
    
    # Create final dataframe with all required columns
    new_df = data[[
        'course_name',
        'University',
        'Difficulty Level',
        'Course Rating',
        'Course URL',
        'Course Description',
        'Skills',
        'tags'
    ]].copy()
    
    return new_df

def create_similarity_matrix(df):
    # Vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    # Calculate similarity
    similarity = cosine_similarity(vectors)
    return similarity

def save_resources(df, similarity):
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    # Save resources
    pickle.dump(similarity, open('models/similarity.pkl', 'wb'))
    pickle.dump(df.to_dict(), open('models/course_list.pkl', 'wb'))
    df.to_pickle('models/courses.pkl')

def main():
    # Preprocess data
    print("Preprocessing data...")
    processed_df = preprocess_data()
    # Create similarity matrix
    print("Creating similarity matrix...")
    similarity_matrix = create_similarity_matrix(processed_df)
    # Save resources
    print("Saving resources...")
    save_resources(processed_df, similarity_matrix)
    print("All resources saved successfully!")

if __name__ == "__main__":
    main()
