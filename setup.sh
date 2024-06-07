#!/bin/bash

# Pull the data from DVC remote storage
dvc pull

# Run the Streamlit app
streamlit run streamlit.py
