ssh -N -f -R 8501:localhost:8501 deevyanshumalu@boa.cse.iitb.ac.in
streamlit run streamlit.py --server.address localhost --server.port 8501