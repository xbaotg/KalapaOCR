import pandas as pd
import pickle


df = pd.read_excel('data/country_fix.xls')
df = df.rename(columns={
    'Tỉnh Thành Phố': 'TTP',
    'Mã TP': 'MTP',
    'Quận Huyện': 'QH',
    'Mã QH': 'MQH',
    'Phường Xã': 'PX',
    'Mã PX': 'MPX',
    'Cấp': 'C',
    'Tên Tiếng Anh': 'TTA'
})
df = df.drop(columns=['C', 'TTA', 'MPX', 'MQH', 'MTP'])
df['TTP'] = df['TTP'].str.replace("-", "").str.replace("  ", " ")
df['PX'] = df['PX'].str.replace("-", " ").str.replace("  ", " ")
df['QH'] = df['QH'].str.replace("-", " ").str.replace("  ", " ")

temp = df[(df['TTP'] == 'Tỉnh Đắk Lắk') & (df['QH'] == 'Huyện Krông A Na')]
temp['QH'] = 'Huyện Krông Ana'
df = pd.concat([df, temp])

temp = []
for row in df.iterrows():
    ttp = row[1]['TTP']
    qh = row[1]['QH']
    px = row[1]['PX']

    if (str(qh) != "nan" and "'" in qh) or (str(ttp) != "nan" and "'" in ttp) or (str(px) != "nan" and "'" in px):
        if "Đắk Nông" in str(ttp):
            print(qh, ttp, px)

        qh = qh.replace("'", "")
        ttp = ttp.replace("'", "")
        px = px.replace("'", "")
        
        # append to df
        temp.append([ttp, qh, px])
        print(temp[-1])

temp = pd.DataFrame(temp, columns=['TTP', 'QH', 'PX'])
df = pd.concat([df, temp], ignore_index=True) 


df_latin = df.copy()
df_latin['TTP'] = df_latin['TTP'].str.replace('Thành phố ', '').str.replace('Tỉnh ', '').str.replace('Thành Phố ', '')
df_latin['QH'] = df_latin['QH'].str.replace('Huyện ', '').str.replace('Thành phố ', '').str.replace('Thị xã ', '').str.replace('Thành Phố ', '').str.replace('Thị Xã ', '')
df_latin['PX'] = df_latin['PX'].str.replace('Xã ', '').str.replace('Thị trấn ', '').str.replace('Thị xã ', '').str.replace('Thị Trấn ', '').str.replace('Thị Xã ', '')

for i, row in df_latin.iterrows():
    if 'Quận' in row['QH']:
        t = row['QH'].replace('Quận ', '')

        if not t.isdigit():
            df_latin.at[i, 'QH'] = t
        else:
            df_latin.at[i, 'QH'] = 'Quận ' + str(int(t))

    if 'Phường' in str(row['PX']):
        t = row['PX'].replace('Phường ', '')

        if not t.isdigit():
            df_latin.at[i, 'PX'] = t
        else:
            df_latin.at[i, 'PX'] = 'Phường ' + str(int(t))


df = pd.concat([df, df_latin])
df.to_csv('data/country_processed.csv', index=False)


# -------------------------------------------------------------------------
df_latin.to_csv('data/country_no_prefix.csv', index=False)


# -------------------------------------------------------------------------
data = {}
data['TTP'] = set(df['TTP'].unique().tolist())
data['QH'] = set(df['QH'].unique().tolist())
data['PX'] = set(df['PX'].unique().tolist())
pickle.dump(data, open('data/country_unique.pkl', 'w+b'))


# -------------------------------------------------------------------------
data = {}

for row in df.itertuples():
    if row.TTP not in data:
        data[row.TTP] = {}

    if row.QH not in data[row.TTP]:
        data[row.TTP][row.QH] = []

    data[row.TTP][row.QH].append(row.PX)

print("data['Khánh Hòa'] = ", data['Khánh Hòa'])

pickle.dump(data, open('data/country_map.pkl', 'w+b'))


# -------------------------------------------------------------------------
data = {}

for row in df_latin.itertuples():
    if row.TTP not in data:
        data[row.TTP] = {}

    if row.QH not in data[row.TTP]:
        data[row.TTP][row.QH] = []

    data[row.TTP][row.QH].append(row.PX)

print("data['Khánh Hòa'] = ", data['Khánh Hòa'])

pickle.dump(data, open('data/country_map_no_prefix.pkl', 'w+b'))
