from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import streamlit as st


@st.cache_resource
def load_fake_news_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "../Models/fake_news_model/distilbert_fakenews_model"
    )
    tokenizer = DistilBertTokenizer.from_pretrained(
        "../Models/fake_news_model/distilbert_fakenews_tokenizer"
    )
    print("Loaded Fake News Model Label Map:", model.config.id2label)
    print("Loaded Fake News Label2ID:", model.config.label2id)


    return model, tokenizer


@st.cache_resource
def load_stance_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        "../Models/stance_detection_model/distilbert_stanceD"
    )
    tokenizer = DistilBertTokenizer.from_pretrained(
        "../Models/stance_detection_model/distilbert_tokenizer_stanceD"
    )
    return model, tokenizer
