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
        
        with st.expander('MOO visualized',expanded=False):
            st.info('Trying to visualize MOO method here. Two departures have been made from implementation in the project to ease visualization -  \n1) Only two criteria have been considered while doing ranking  \n2) Number of points along each criteria to get convex combination of objectives has been set arbitrarily set to a relatively lower value as H=8 (please refer section 6.6.6.1 of UJ5 SDS for more details on H)')
            cols = df.columns.to_list()
            option1 = st.selectbox('Select two criteria to base your MOO on',
            [[cols[0],cols[1]],[cols[1],cols[2]],[cols[0],cols[2]]])

            if st.button('Run MOO'):
                fig,df = MOO_explain(df[option1])
                st.write(df)
                st.plotly_chart(fig,use_container_width=True)

    