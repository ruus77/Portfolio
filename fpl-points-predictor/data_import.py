import pandas as pd
seasons = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']


def data_import(seasons_list:list):
  lst = []
  for s in seasons_list:
    for gw in range(1, 39):
        try:
            url = f"https://github.com/vaastav/Fantasy-Premier-League/raw/master/data/{s}/gws/gw{gw}.csv"
            df_temp = pd.read_csv(url, encoding="utf-8").assign(season=s, gw=gw)
            if not df_temp.empty:
              lst.append(df_temp)
        except: break
  data = pd.concat(lst, ignore_index=True).copy()
  return data


df_original = data_import(seasons_list=seasons)
df = df_original.copy()
df.to_csv('fpl_data_2021_2526.csv', index=False)

