# new_app1.py
import streamlit as st
import io
from utils import *
import pandas as pd
import seaborn as sns
import plotly.express as px

def app():

    def AHP_rel_imp(df,criteria_matrix):
        method = AHP('Relative Weight',df.shape[1]) # the second argument is number of objectives
        method.update_criteria(list(df.columns))
        method.update_matrix(criteria_matrix)
        return method.rank().T

    st.set_option('deprecation.showfileUploaderEncoding', False)
                   
    st.title('AHP ranking and scheduling')

    st.sidebar.title("Upload file")
    temp = st.sidebar.file_uploader(label='', type=['xlsx'])

    # df = load_input()
    if temp:

        with st.expander('Interventions',expanded=False):
            st.info('Bundling logic:  \n Cost-sum, Risk-max, FPMK-sum, Startdate-min, Enddate-max')
            df = pd.read_excel(temp)
            st.text('Before bundling')
            st.dataframe(df)
            df['Bundle'].fillna(df.index.to_series()+1000, inplace=True) # giving bundle numbers to empty strings, such that groupby works for them
            df = df.groupby('Bundle').aggregate({'Cost':'sum','Risk':'max','FPMK':'sum','Start Date':'min','End Date':'max'}).reset_index(drop=True)
            # st.subheader('Interventions')        
            st.text('After bundling')
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
                
                st.info('AHP:  \nMaximizes delta-FPMK, minizes Risk  \nHighest value of AHP-score --> Rank-1')
                st.info('Cost and AHP based final rank:  \nCost-Benefit Ratio(CBR) = AHP-score/(normalized cost)  \nHighest CBR --> Rank-1')

                if st.button('Rank and schedule'):
                    to_display_df = df.copy()
                    df['Risk'] = 1.0/df['Risk'] #taking inverse of risk, trying to maximize
                    # a = 1.0
                    # a,b,c = 1.0,1.0,1.0
                    criteria_matrix = np.array([[1.0, crit_value],[1.0/crit_value, 1.0]])# np.array([[1.0,a,b],[1/a,1.0,c],[1/c,1/b,1.0]])
                    ahp_df= AHP_rank(df[['Risk','FPMK']],criteria_matrix) 
                    st.write(AHP_rel_imp(df[['Risk','FPMK']],criteria_matrix))
                    # st.write(crit_importance)
                    CBR = ahp_df['AHP Score'].values/(df.Cost/df.Cost.sum()).values # changed for AHP-maximize
                    # CBR = 1/((df.Cost/df.Cost.sum()).values*ahp_df['AHP Score'].values) # computing cost-benefit ratio
                    # CBR = (1-ahp_df['AHP Score'].values)/((df.Cost/df.Cost.sum()).values) # computing cost-benefit ratio
                    rank_ = len(CBR) - np.argsort(np.argsort(CBR))
                    to_display_df['AHP Score'] = ahp_df['AHP Score']     
                    to_display_df['AHP-ranks'] = ahp_df['rank'].apply(np.int64)#.astype(uint8) 
                    to_display_df['Normalized Cost'] =  ((df.Cost/df.Cost.sum()).values)              
                    to_display_df['Benefit Cost Ratio'] = CBR#/(np.sum(CBR))
                    to_display_df['Benefit Cost Ratio-ranks'] = rank_#ahp_df['rank'].apply(np.int64)#.astype(uint8) 
                    to_display_df.drop(columns=['Start Date','End Date'],inplace=True)               
                    cm = sns.light_palette("green", as_cmap=True, reverse=True) #
                    st.dataframe(to_display_df.style.background_gradient(cmap=cm))

                    df['duration-months'] = ((df['End Date'] - df['Start Date'])/np.timedelta64(1, 'M')).astype(int)
                    ints = [Intervention(df['duration-months'][i],(int(df['End Date'][i].year),int(df['End Date'][i].month)),i+1,"test",1) for i in range(df.shape[0])]
                    list_tasks_dates= []

                    count = 1
                    rank_custom = rank_
                    for item in ints:
                        item.add_rank(rank_custom[count-1])
                        count += 1
                    #     print(item.startdate.year_month(),item.enddate.year_month())
                        list_tasks_dates.append(['Original Intervention - ' + str(count-1),str(item.startdate.year_month())+'01',str(item.enddate.year_month())+'28'])

                    df_old = pd.DataFrame(list_tasks_dates,columns = ['Intervention','Start','End'])
                    df_old[['Start','End']] = df_old[['Start','End']].apply(pd.to_datetime)
                    df_old['Schedule'] = 'Original'
                    df_old['Rank'] = ['Rank-'+str(i) for i in rank_custom]

                    facility_intervention = ints
                    supply_num = 1 #assuming each facility can only support one intervention at a time
                    current_num = 0

                    for time in [i for i in range(df['duration-months'].sum()+1)]:
                        
                        current_projects = list(np.array(facility_intervention)[[time in x for x in facility_intervention]]) #current projects
                        starting_projects = list(np.array(facility_intervention)[[time == x.startdate.time for x in facility_intervention]])    
                        local_ranks = [x.rank for x in starting_projects]
                        
                        current_num = np.array([x.num for x in current_projects]).sum()
                        starting_num = np.array([x.num for x in starting_projects]).sum()
                        
                    #     if time%12 == 1:
                    #         print(time//12 + 2020)
                    #         print([(int(x.rank),x.startdate.year_month(),x.enddate.year_month()) for x in current_projects])
                            
                        while current_num > supply_num:
                            local_max = np.array(local_ranks).argmax() # max rank of starting-projects, lowest priority intervention
                            current_num -= starting_projects[local_max].num # decrement current_num
                            starting_projects[local_max].change() #change occurs here, starting proj with highest rank (lowest priority) is changed
                            # what is the change? --> startdate += 1 and duration is shortened
                            local_ranks.pop(local_max)
                            starting_projects.pop(local_max)

                    list_tasks_dates= []
                    count = 1
                    for item in ints:
                        count += 1
                    #     print(item.startdate.year_month(),item.enddate.year_month())
                        list_tasks_dates.append(['Deconflicted Intervention - ' + str(count-1),str(item.startdate.year_month())+'01',str(item.enddate.year_month())+'28'])

                    df_new = pd.DataFrame(list_tasks_dates,columns = ['Intervention','Start','End'])
                    df_new[['Start','End']] = df_new[['Start','End']].apply(pd.to_datetime)
                    df_new['Schedule'] = 'Deconflicted'
                    df_new['Rank'] = ['Rank-'+str(i) for i in rank_custom]

                    df_combined = pd.concat([df_old,df_new])
                    fig = px.timeline(df_combined, x_start="Start", x_end="End", y="Intervention",color="Schedule",text='Rank')
                    fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
                    fig.update_xaxes(title_text='Year')
                    st.plotly_chart(fig)

