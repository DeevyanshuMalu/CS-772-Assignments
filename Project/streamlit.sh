curl --location-trusted -u 22b1029:f07b3c72a55d9598cc17fa115061476c "https://internet-sso.iitb.ac.in/login.php"
unset HF_HUB_CACHE

ssh -N -f -R 8501:localhost:8501 ${USER}@boa.cse.iitb.ac.in
streamlit run streamlit.py --server.address localhost --server.port 8501