import streamlit as st
import pandas as pd, numpy as np
import pymc as pm

from streamlit_gsheets import GSheetsConnection

reports_sheet_url = "https://docs.google.com/spreadsheets/d/1STDQHopa5_gagC4q4iHe2vku-YQvVROxOa89dOrBgJc/edit?usp=sharing"

is_admin = (st.query_params.get('admin') == st.secrets['admin_password'])

# Create a connection object.
@st.cache_resource
def get_gs_conn():
    return st.connection("gsheets", type=GSheetsConnection)
conn = get_gs_conn()

@st.cache_data(ttl='10min')
def get_result_data():
    # Get battle reports
    frdf = conn.read(worksheet="Responses",spreadsheet=reports_sheet_url)
    rdf = frdf[['Your username', 'Opponent username', 'Who won?', 'When did the game take place?','Format']]
    rdf.columns = ['Player 1','Player 2','Result','Time','Format']
    rdf['Time'] = pd.to_datetime(rdf['Time'],format='mixed')
    rdf = rdf[rdf['Result'].isin(['Opponent','Tie','Me'])]
    rdf['SKOOR_A'] = rdf['Result'].replace({'Opponent':0,'Tie':0.5,'Me':1}).astype('float')
    rdf['SKOOR_B'] = rdf['Result'].replace({'Opponent':1,'Tie':0.5,'Me':0}).astype('float')
    df = rdf.drop(columns=['Result'])

    # Convert usernames to lowercase
    df['Player 1'] = df['Player 1'].str.strip().str.lower()
    df['Player 2'] = df['Player 2'].str.strip().str.lower()

    # Replace aliases
    aliases = conn.read(worksheet="Aliases",spreadsheet=reports_sheet_url)
    #aliases = sh.worksheet('Aliases').get_all_records()
    amap = { v['Alias'].lower(): v['Username'].lower() for i,v in aliases.iterrows()}
    df[['Player 1','Player 2']] = df[['Player 1','Player 2']].replace(amap)

    return df

@st.cache_data(ttl='1h')
def compute_elo(game_type):
    df = get_result_data()

    if game_type!=None:
        df = df[df['Format']==game_type]
    df = df.drop(columns=['Format'])

    # Create a list of all usernames
    players = list(set(df['Player 1'].unique()) | set(df['Player 2']))

    # Convert usernames to indices for model
    p1i = df['Player 1'].apply(lambda p: players.index(p))
    p2i = df['Player 2'].apply(lambda p: players.index(p))

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
    gcounts = pd.concat([df['Player 1'],df['Player 2']]).value_counts()

    # Get wins/ties/losses
    wins = pd.concat([df[df['SKOOR_A']==1]['Player 1'],df[df['SKOOR_B']==1]['Player 2']]).value_counts()
    losses = pd.concat([df[df['SKOOR_A']==0]['Player 1'],df[df['SKOOR_B']==0]['Player 2']]).value_counts()
    ties = pd.concat([df[df['SKOOR_A']==0.5]['Player 1'],df[df['SKOOR_B']==0.5]['Player 2']]).value_counts()
    wltdf = pd.DataFrame({'w':wins, 'l':losses, 't':ties}).fillna(0).astype('int')

    # Compile a result dataframe
    res_df = pd.DataFrame({'ELO':ranking.round(0).astype('int'),'Games':gcounts, 
                            'Wins': wltdf['w'], 'Losses': wltdf['l'], 'Ties':wltdf['t']}).sort_values('ELO',ascending=False)
    res_df['Username'] = res_df.index

    return res_df


@st.cache_data(ttl='1min')
def fetch_public():
    return list(conn.read(worksheet="Public",spreadsheet=reports_sheet_url)['Username'])

# Filter only those that have given permission
public = { u.lower(): u for u in fetch_public() }

formats = [None, '2000 pts'] if not is_admin else [None]

tabs = st.tabs([ f if f is not None else 'Kõik' for f in formats])

for ti, stt in enumerate(tabs):
    res_df = compute_elo(formats[ti])
    total_games = res_df['Games'].sum()//2

    if not is_admin: # For regular users, show only public usernames
        res_df = res_df[res_df.index.isin(public) | (res_df['Games']>=3)]
        res_df.loc[~res_df.index.isin(public),'Username'] = '-'
        res_df.loc[~res_df.index.isin(public) & (res_df['Games']>=5),'Games'] = '5+'
        res_df.loc[~res_df.index.isin(public),['Wins','Losses','Ties']] = ''
        res_df = res_df[['Username','Games','Wins','Losses','Ties','ELO']]
    else: # For admins, show unfiltered full list
        res_df['Public'] = res_df.index.isin(public)
        res_df = res_df[['Public','Username','Games','Wins','Losses','Ties','ELO']]

    res_df.index = range(1,len(res_df)+1)
    res_df['Username'] = res_df['Username'].replace(public)

    stt.markdown(f'''
    # Adeptus Estonicus W40k ranking
    Based on {total_games} games, mostly those reported [here](https://forms.gle/43u8m5WSsJhqFrbJ8).  
    PM *@velochy2* (Margus) in Discord if you want your name visible
    ''')

    # Add some admin tools to help manage the spreadsheet
    if is_admin:
        stt.header("Admin tools")
        stt.markdown(f"[Link to spreadsheet]({reports_sheet_url})")

        if stt.button("Force recompute"):
            st.cache_data.clear()
        
        stt.header("Full ranking")

    stt.dataframe(res_df,use_container_width=True,height=50+len(res_df)*35)

    if is_admin:
        stt.header("Duplicate games")
        from collections import defaultdict
        rdf = get_result_data()
        games, row_ids = defaultdict(list), defaultdict(list) 
        for i,r in rdf.iterrows():
            pt = tuple({r['Player 1'],r['Player 2']}) # This makes sure the pair is always ordered same way
            p1s = r['SKOOR_A'] if pt[0]==r['Player 1'] else r['SKOOR_B'] # Score of the first player in tuple
            games[pt + (p1s,)].append(r['Time'])
            row_ids[pt + (p1s,)].append(i+2)

        for k,l in games.items():
            if len(l)<=1: continue
            l = list(pd.Series(l).sort_values())
            for i, v in enumerate(l[:-1]):
                if (l[i+1]-v)<=pd.Timedelta('2d'):
                    stt.write(f"Potential duplicate: {k}, {row_ids[k][i]}, {v}, {row_ids[k][i+1]}")
            #st.write(k,l)

        stt.header("Most similar usernames")
        from itertools import combinations
        from textdistance import strcmp95
        new_df = pd.DataFrame(combinations(res_df['Username'], 2), columns=["id1","id2"])
        new_df["EDist"] = new_df.apply(lambda x: strcmp95(x[0].lower(),x[1].lower()), axis=1)
        stt.dataframe(new_df.sort_values('EDist',ascending=False)[:30])

