from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

import streamlit as st
import base64

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

dataset = pd.read_csv("reduced_regional.csv")
dataset['features'] = dataset['Genre'] + ' ' + dataset['Language'] + ' ' + dataset['Votes']
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['features'])

knn_model = joblib.load('knn_model.pkl')
def get_recommendations(query):
    query_vector = tfidf_vectorizer.transform([query])
    distances, indices = knn_model.kneighbors(query_vector)
    recommendations = dataset.iloc[indices[0]].copy()  # Copy the DataFrame slice
    return recommendations

# def highlight_high_rating(x):
#         return ['background-color: #00FF00' if val >= 5.0 else 'background-color: #87CEEB' for val in x]
    
def main():
    st.set_page_config(page_title="Movie Recommendation",page_icon="🍿")
    set_background('bgimage2.jpg')
    st.markdown("<h1 style='color: #FF5733;'>Indian Movie Recommendation System 🍿</h1>", unsafe_allow_html=True)
    option1 = st.selectbox('Select an option', ['assamese', 'bengali', 'bhojpuri','gujarati', 'hindi', 'kannada','kashmiri', 'konkani', 'malayalam','marathi', 'nepali', 'oriya'
                                                        'punjabi', 'rajastani', 'sanskrit','tamil', 'telugu', 'tulu','urdu'])
    
    option2 = st.selectbox('Select an option', ['Action', 'Adventure','Animation','Comedy', 'Crime','Drama', 'Documentary', 'Mystery','Short', 'History', 'Music','Romance', 'Thriller', 'Horror'
                                                'Family', 'Biography', 'Sport'])
    
    option = option1+" "+option2
    submit=st.button("Recommend")
    if submit:   
        recommendations=get_recommendations(option)
        st.markdown("<h2 style='color: #3366FF;'>The Movies are:</h2>", unsafe_allow_html=True)
        # st.write(recommendations)

        # Apply the styling to the DataFrame
        # styled_df = recommendations.style.apply(highlight_high_rating, subset=['Rating(10)'])

        # Display the interactive DataFrame
        st.write(recommendations)
        # for rating in recommendations['Movie Name']:
        # markdown_str = f'<input type="range" min="0" max="10" value="{rating}" step="0.1" oninput="result.value=this.value">'
        # markdown_str += '<output name="result"></output>'
        # st.markdown(markdown_str, unsafe_allow_html=True)
        # st.write(recommendations[['Movie Name','Rating(10)','Votes',]])
        # st.write(
        #     f'<style>.dataframe {{ width: 800px; height: 200px; }}</style>',
        #     unsafe_allow_html=True
        #     )
        # st.dataframe(recommendations[['Movie Name','Rating(10)','Votes',]])
        # Create a Streamlit app
        # st.write(f"Recommendations for query: {option}")

        # Plotting the recommendations
        fig, ax = plt.subplots(figsize=(12, 8))

        # Bar plot
        bars = ax.bar(recommendations['Language'] + ': ' + recommendations['Movie Name'], recommendations['Rating(10)'], color='skyblue')

        # Adding rating as annotations on the bars
        for bar, rating in zip(bars, recommendations['Rating(10)']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{rating:.2f}', ha='center', color='black', fontsize=10)

        ax.set_xlabel('Language: Movie Name')
        ax.set_ylabel('Rating(10)')
        ax.set_title('Recommendations for query: ' + option)
        ax.tick_params(axis='x', rotation=45, labelsize=10)

        # Display the Matplotlib plot within the Streamlit app
        st.pyplot(fig)

if __name__ == "__main__":
    main()

