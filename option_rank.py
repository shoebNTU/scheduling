# new_app1.py
import pandas as pd
import streamlit as st
import io

from utils import MOO_explain

def app():                
    st.title('MOO')

    st.sidebar.title("Upload file")
    temp = st.sidebar.file_uploader(label='', type=['xlsx'])

    # df = load_input()
    if temp:
        
        with st.expander('Options',expanded=False):
            df = pd.read_excel(temp)
            st.dataframe(df.iloc[:,:3])
        
        with st.expander('MOO',expanded=False):
            cols = df.columns.to_list()
            option1 = st.selectbox('Select two criteria to base your MOO on',
            [[cols[0],cols[1]],[cols[1],cols[2]],[cols[0],cols[2]]])

            if st.button('Run MOO'):
                fig,df = MOO_explain(df[option1])
                st.write(df)
                st.plotly_chart(fig,use_container_width=True)

    