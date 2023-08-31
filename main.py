import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    .header { font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    .container { padding: 20px; border: 1px solid #ddd; border-radius: 10px; margin-top: 20px; }
    .expander { background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px; }
    .big-header { font-size: 30px; font-weight: bold; margin-top: 20px; }
    .comment-container { display: flex; justify-content: space-between; margin-top: 20px; }
    .comment { width: 48%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

model_name = "textattack/roberta-base-SST-2"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

def scrape_amazon_reviews(url, num_pages=1):
    all_reviews = []
    for page in range(1, num_pages + 1):
        response = requests.get(url + f'?pageNumber={page}')
        soup = BeautifulSoup(response.content, 'html.parser')
        reviews = []
        for review in soup.find_all('span', class_='a-size-base review-text review-text-content'):
            reviews.append(review.get_text())
        all_reviews.extend(reviews)
    return all_reviews

def get_recommendation(total_sentiment_score):
    if total_sentiment_score < 0.2:
        return "Strongly Not Recommended 👎"
    elif 0.2 <= total_sentiment_score < 0.4:
        return "Not Recommended 👎"
    elif 0.4 <= total_sentiment_score < 0.6:
        return "Neutral 😐"
    elif 0.6 <= total_sentiment_score < 0.8:
        return "Recommended 👍"
    else:
        return "Strongly Recommended 👍👍"

st.title("Online Product Review Analysis")
st.markdown("Analyzing the sentiment of E-commerce website product reviews")

# Sidebar navigation menu
page = st.sidebar.selectbox("Need a guide? Switch to that Page!", ["Home", "User Guide"])

if page == "Home":
    amazon_url = st.text_input("Enter Amazon Product Review URL:", key="home_amazon_url")
    num_pages = st.number_input("Enter Number of Pages to Scrape:", min_value=1, value=1, key="home_num_pages")

    if st.button("Scrape and Analyze Reviews", key="home_analyze_button"):
        if amazon_url:
            st.info("Fetching and analyzing reviews...")
            reviews = scrape_amazon_reviews(amazon_url, num_pages)
            reviews_df = pd.DataFrame({'Text': reviews})
            reviews_df.to_csv('amazon_reviews.csv', index=False)
            st.success(f"{len(reviews)} reviews scraped and saved to 'amazon_reviews.csv'")

            total_sentiment_score = 0
            sentiment_scores = []

            for i, review in enumerate(reviews_df['Text']):
                review_text = str(review)
                inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits).item()
                sentiment_score = 1.0 if predicted_class == 1 else 0.0
                sentiment_scores.append(sentiment_score)
                total_sentiment_score += sentiment_score

            overall_recommendation = get_recommendation(total_sentiment_score)

            if len(reviews_df) > 0:
                overall_sentiment_percentage = (total_sentiment_score / len(reviews_df)) * 100
                st.markdown("<p class='big-header'>Overall Sentiment Score (%): " + 
                            str(round(overall_sentiment_percentage, 2)) + "</p>", unsafe_allow_html=True)
            else:
                st.warning("No reviews available.")

            st.write("Overall Recommendation:", overall_recommendation)

            if sentiment_scores:
                best_index = sentiment_scores.index(max(sentiment_scores))
                worst_index = sentiment_scores.index(min(sentiment_scores))
                positive_reviews = sum(sentiment_score > 0 for sentiment_score in sentiment_scores)
                negative_reviews = len(sentiment_scores) - positive_reviews

                fig, ax = plt.subplots()
                ax.bar(["Positive", "Negative"], [positive_reviews, negative_reviews], color=['green', 'red'])
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Number of Reviews")
                ax.set_title("Distribution of Positive and Negative Reviews")
                st.pyplot(fig)

                st.markdown("---")
                st.markdown("<div class='comment-container'>", unsafe_allow_html=True)
                st.markdown("<div class='comment'>", unsafe_allow_html=True)
                st.markdown("### Best Comment")
                st.write(reviews_df['Text'][best_index])
                st.write("Sentiment Score (%):", round(sentiment_scores[best_index] * 100, 2))
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='comment'>", unsafe_allow_html=True)
                st.markdown("### Worst Comment")
                st.write(reviews_df['Text'][worst_index])
                st.write("Sentiment Score (%):", round(sentiment_scores[worst_index] * 100, 2))
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

elif page == "User Guide":
    st.title("User Guide")
    st.write("Welcome to the User Guide page!")

    st.write("Open your Product Page and click on Ratings:")
    st.image("picture1.png", use_column_width=True)

    st.write("Scroll down till you find this and click it:")
    st.image("picture2.png", use_column_width=True)

    st.write("Copy the URL of this page and paste it into this website for scraping:")
    st.image("picture3.png", use_column_width=True)
