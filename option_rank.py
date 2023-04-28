# new_app1.py
import pandas as pd
import streamlit as st
from utils import MOO_explain, MOO_explain_3d
from itertools import combinations

def app():                
    st.title('Multi-objective Optimization (MOO)')

    st.sidebar.title("Upload file")
    temp = st.sidebar.file_uploader(label='', type=['xlsx'])

    # df = load_input()
    if temp:
        
        with st.expander('Options',expanded=False):
            df = pd.read_excel(temp)
            cols = [col for col in df.columns if col.lower().split(' ')[0] not in ['bundle', 'start','end']]
            df = df[cols]
            df = df.select_dtypes(exclude='datetime64[ns]') #remove datetime type
            st.dataframe(df)

        with st.expander('Criteria',expanded=True):
            criteria = st.multiselect(
                'What criteria do you want to use for comparison? Please select at least 2 criteria.',
                df.columns.to_list(),df.columns.to_list())

            if len(criteria)>1:
                crit_min_max = []
                for crit in criteria:                    
                    min_max =  st.selectbox(f'Do you wish to minimize or maximize {crit}',('Minimize','Maximize'))
                    crit_min_max.append(min_max)
                df = df[list(criteria)]
            else:
                st.error("Please select at least 2 or more criteria.")
        
        with st.expander('MOO visualized',expanded=True):

            if len(criteria)>1:

                st.info('Visualizing MOO method here')
                cols = df.columns.to_list()

                if len(cols)>3:
                    st.info('We cannot visualize beyond three criteria. \nHence, limiting criteria to three.')
                    option1 = st.selectbox('Select three criteria to base your MOO on',list(map(list,list(combinations(df.columns, 3)))))
                else:
                    option1 = list(df.columns)

                if len(option1) == 2:
                    if st.button('Run MOO',key=456):
                        fig,df = MOO_explain(df[option1],crit_min_max)
                        st.write(df)
                        st.plotly_chart(fig,use_container_width=True)
                else:
                    if st.button('Run MOO',key=345):
                        fig,df = MOO_explain_3d(df[option1],crit_min_max)
                        st.write(df)
                        st.plotly_chart(fig,use_container_width=True)

            else:
                st.error("Please select at least 2 or more criteria.")
