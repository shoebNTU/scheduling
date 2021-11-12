# new_app1.py
import streamlit as st
import io
from utils import *
import pandas as pd
import seaborn as sns


st.set_option('deprecation.showfileUploaderEncoding', False)

with open('favicon.png', 'rb') as f:
    favicon = io.BytesIO(f.read())

st.set_page_config(page_title='Ranking Interventions',
                   page_icon=favicon, 
                   layout='wide', 
                   initial_sidebar_state='expanded')
                
st.title('Ranking Interventions')

st.sidebar.title("Upload file")
temp = st.sidebar.file_uploader(label='', type=['xlsx'])

# df = load_input()
if temp:

    with st.expander('Interventions',expanded=False):
        df = pd.read_excel(temp)
        # st.subheader('Interventions')
        st.dataframe(df)
    # df = pd.read_excel('ip_1.xlsx') # reading inputs

    # if st.button('Show interventions'):
    #     st.subheader('Interventions')
    #     st.write(df)
    with st.expander('Ranking',expanded=False):
        if st.button('Show ranking'):
            to_display_df = df.copy()
            to_display_df['MOO-ranks'] = MOO(df)
            a,b,c = 1.0,1.0,1.0
            criteria_matrix = np.array([[1.0,a,b],[1/a,1.0,c],[1/c,1/b,1.0]])
            ahp_df = AHP_rank(df,criteria_matrix)            
            to_display_df['AHP-ranks'] = ahp_df['rank'].apply(np.int64)#.astype(uint8)
            to_display_df['Michael\'s-ranks'] = sum_of_rank(df)            
            # topsis_rank, simus_rank = TOPSIS_SIMUS(df)
            topsis_rank = TOPSIS_SIMUS(df)
            to_display_df['TOPSIS'] = pd.Series(topsis_rank).apply(np.int64)
            # to_display_df['SIMUS'] = pd.Series(simus_rank).apply(np.int64)

            cm = sns.light_palette("green", as_cmap=True, reverse=True)
            st.dataframe(to_display_df.style.background_gradient(cmap=cm))
