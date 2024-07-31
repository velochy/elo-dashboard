import streamlit as st
import pandas as pd, numpy as np
import pymc as pm

from streamlit_gsheets import GSheetsConnection

reports_sheet_url = "https://docs.google.com/spreadsheets/d/1STDQHopa5_gagC4q4iHe2vku-YQvVROxOa89dOrBgJc/edit?usp=sharing"

# Create a connection object.
@st.cache_resource
def get_gs_conn():
    return st.connection("gsheets", type=GSheetsConnection)
conn = get_gs_conn()

@st.cache_data(ttl='1h')
def compute_elo():

    # Get battle reports
    frdf = conn.read(worksheet="Responses",spreadsheet=reports_sheet_url)
    rdf = frdf[['Sinu kasutajanimi', 'Vastase kasutajanimi', 'Kes võitis?']]
    rdf.columns = ['Mängija A','Mängija B','Tulemus']
    rdf = rdf[rdf['Tulemus'].isin(['Vastane','Viik','Mina'])]
    rdf['SKOOR_A'] = rdf['Tulemus'].replace({'Vastane':0,'Viik':0.5,'Mina':1}).astype('float')
    rdf['SKOOR_B'] = rdf['Tulemus'].replace({'Vastane':1,'Viik':0.5,'Mina':0}).astype('float')
    rdf.drop(columns=['Tulemus'],inplace=True)
    df = rdf

    # Convert usernames to lowercase
    df['Mängija A'] = df['Mängija A'].str.strip().str.lower()
    df['Mängija B'] = df['Mängija B'].str.strip().str.lower()

    # Replace aliases
    aliases = conn.read(worksheet="Aliases",spreadsheet=reports_sheet_url)
    #aliases = sh.worksheet('Aliases').get_all_records()
    amap = { v['Alias'].lower(): v['Username'].lower() for i,v in aliases.iterrows()}
    df[['Mängija A','Mängija B']] = df[['Mängija A','Mängija B']].replace(amap)

    # Create a list of all usernames
    players = list(set(df['Mängija A'].unique()) | set(df['Mängija B']))

    # Convert usernames to indices for model
    p1i = df['Mängija A'].apply(lambda p: players.index(p))
    p2i = df['Mängija B'].apply(lambda p: players.index(p))

    # Convert scores to integers (0/0.5/1 to 0/1/2)
    p1r = (df['SKOOR_A']*2).astype('int')

    # Model ELO
    elo_mean, elo_sd = 1000, 1000
    scale = np.log(10)/400
    with pm.Model(coords={ 'players': players, 'matches': np.arange(len(df)) }) as model:
        elos = pm.Normal('elos',elo_mean,elo_sd,dims=['players'])
        e_scaled = elos*scale

        cp = pm.HalfNormal('tie_range',1)

        # Give everyone a tie against the "average" player to start out
        pm.OrderedLogistic('norm', e_scaled-elo_mean*scale, cutpoints=[-cp,cp], dims=['players'], observed=np.ones(len(players),dtype='int'))
        
        pm.OrderedLogistic('results', e_scaled[p1i]-e_scaled[p2i], cutpoints=[-cp,cp], dims=['matches'], observed=p1r)

    # Run full bayes model
    #with model:
    #    idata = pm.sample(nuts_sampler='numpyro')
    #ranking = pd.Series(idata.posterior.elos.mean(['chain','draw']),index=players).sort_values()

    # Just find MAP (faster)
    with model:
        mres = pm.find_MAP()
    ranking = pd.Series(mres['elos'],index=players).sort_values()

    # Get game counts
    gcounts = pd.concat([df['Mängija A'],df['Mängija B']]).value_counts()

    # Get wins/ties/losses
    wins = pd.concat([df[df['SKOOR_A']==1]['Mängija A'],df[df['SKOOR_B']==1]['Mängija B']]).value_counts()
    losses = pd.concat([df[df['SKOOR_A']==0]['Mängija A'],df[df['SKOOR_B']==0]['Mängija B']]).value_counts()
    ties = pd.concat([df[df['SKOOR_A']==0.5]['Mängija A'],df[df['SKOOR_B']==0.5]['Mängija B']]).value_counts()
    wltdf = pd.DataFrame({'w':wins, 'l':losses, 't':ties}).fillna(0).astype('int')

    # Compile a result dataframe
    res_df = pd.DataFrame({'ELO':ranking.round(0).astype('int'),'Games':gcounts, 
                            'Wins': wltdf['w'], 'Losses': wltdf['l'], 'Ties':wltdf['t']}).sort_values('ELO',ascending=False)
    res_df['Username'] = res_df.index

    return res_df

res_df = compute_elo()

@st.cache_data(ttl='1min')
def fetch_public():
    return list(conn.read(worksheet="Public",spreadsheet=reports_sheet_url)['Username'].str.lower())

# Filter only those that have given permission
public = fetch_public()

#public = [p['Username'].lower() for p in sh.worksheet('Public').get_all_records()]
total_games = res_df['Games'].sum()//2
res_df = res_df[res_df.index.isin(public) | (res_df['Games']>=3)]
res_df.loc[~res_df.index.isin(public),'Username'] = '-'
res_df.loc[~res_df.index.isin(public) & (res_df['Games']>=5),'Games'] = '5+'
res_df.loc[~res_df.index.isin(public),['Wins','Losses','Ties']] = ''
res_df.index = range(1,len(res_df)+1)

st.markdown(f'''
# Adeptus Estonicus W40k ranking
Based on 52 games scraped by Metsawend from #lahingumöllud + {total_games-52} games reported [here](https://forms.gle/43u8m5WSsJhqFrbJ8)  
PM *@velochy2* (Margus) in Discord if you want your name visible
''')
st.dataframe(res_df[['Username','Games','Wins','Losses','Ties','ELO']],use_container_width=True,height=50+len(res_df)*35)
