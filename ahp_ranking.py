# new_app1.py
import streamlit as st
import io
from utils import *
import pandas as pd
import seaborn as sns

def AHP_rel_imp(df,criteria_matrix):
    method = AHP('Relative Weight',df.shape[1]) # the second argument is number of objectives
    method.update_criteria(list(df.columns))
    method.update_matrix(criteria_matrix)
    return method.rank().T

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

        option1 = st.selectbox('What is more important',['Risk','FPMK','Neither of the above'])
        
        if option1 == 'Risk':
            option = st.selectbox('How important is Risk against FPMK',['1 (Equally important)','3 (Somewhat more important)',
            '5 (Much more important)','7 (Very much more important)', '9 (Absolutely more important)'])
            crit_value = int(option[0])
        elif option1 == 'FPMK':
            option = st.selectbox('How important is FPMK against Risk',['1 (Equally important)','3 (Somewhat more important)',
            '5 (Much more important)','7 (Very much more important)', '9 (Absolutely more important)'])
            crit_value = 1.0/int(option[0])
        else:
           crit_value = 1.0        
        
        if crit_value:
            if st.button('Show ranking'):
                to_display_df = df.copy()
                # a = 1.0
                # a,b,c = 1.0,1.0,1.0
                criteria_matrix = np.array([[1.0, crit_value],[1.0/crit_value, 1.0]])# np.array([[1.0,a,b],[1/a,1.0,c],[1/c,1/b,1.0]])
                ahp_df= AHP_rank(df[['Risk','FPMK']],criteria_matrix) 
                st.write(AHP_rel_imp(df[['Risk','FPMK']],criteria_matrix))
                # st.write(crit_importance)
                # CBR = 1/((df.Cost/df.Cost.sum()).values*ahp_df['AHP Score'].values) # computing cost-benefit ratio
                CBR = (1-ahp_df['AHP Score'].values)/((df.Cost/df.Cost.sum()).values) # computing cost-benefit ratio
                rank_ = len(CBR) - np.argsort(np.argsort(CBR))
                to_display_df['AHP Score'] = ahp_df['AHP Score']     
                to_display_df['AHP-ranks'] = ahp_df['rank'].apply(np.int64)#.astype(uint8) 
                to_display_df['Normalized Cost'] =  ((df.Cost/df.Cost.sum()).values)              
                to_display_df['Cost Benefit Ratio'] = CBR#/(np.sum(CBR))
                to_display_df['Cost Benefit Ratio-ranks'] = rank_#ahp_df['rank'].apply(np.int64)#.astype(uint8)                
                cm = sns.light_palette("green", as_cmap=True, reverse=True)
                st.dataframe(to_display_df.style.background_gradient(cmap=cm))
