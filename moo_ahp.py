# new_app1.py
from multiapp import MultiApp
import streamlit as st
import io

# import app
import schedule_pr
import option_rank

st.set_option('deprecation.showfileUploaderEncoding', False)

# favicon = open("favicon.png", "rb", buffering=0)
with open('favicon.png', 'rb') as f:
    favicon = io.BytesIO(f.read())

st.set_page_config(page_title='UJ5 Optimization',
                   page_icon=favicon, 
                   layout='wide', 
                   initial_sidebar_state='expanded')

app = MultiApp()
app.add_app("Option ranking", option_rank.app)
app.add_app("Schedule Prioritization", schedule_pr.app)


app.run()
