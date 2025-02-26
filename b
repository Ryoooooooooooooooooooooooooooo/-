import pandas as pd

# CSVファイルの読み込み
file_path = '/mnt/data/鳥インフルエンザ1.csv'
data = pd.read_csv(file_path)

# データの先頭部分を表示して内容を確認
data.head()

import re

# 日付の整形
data['発生日'] = pd.to_datetime(data['発生日'].str.replace('令和６', '令和6').str.replace('令和', '').str.replace('年', '-').str.replace('月', '-').str.replace('日', '').str.strip(), era='western')

# 飼養羽数の数値化
def parse_birds(s):
    s = s.replace('羽', '').replace('万', 'e4').strip()
    return pd.to_numeric(s, errors='coerce')

data['飼養羽数'] = data['飼養羽数'].apply(parse_birds)

# 不要なカラムの削除
data = data[['発生場所', '発生日', '飼養羽数']]

# 緯度と経度のダミーデータを追加（本来は具体的な位置データが必要）
# ここでは日本の主要都市の緯度経度を例示的に使用
coordinates = {
    '北海道厚真町': (42.5067, 141.9064),
    '千葉県香取市': (35.8972, 140.4996),
    '新潟県上越市': (37.1483, 138.2364),
    '島根県大田市': (35.1973, 132.4999),
    '新潟県胎内市': (37.6691, 139.0469)
}

data['latitude'] = data['発生場所'].map(lambda x: coordinates.get(x, (None, None))[0])
data['longitude'] = data['発生場所'].map(lambda x: coordinates.get(x, (None, None))[1])

# 変換後のデータを確認
data.head()

# 日付の元号を西暦に変換する関数
def convert_era_to_western(date_str):
    era_map = {
        '令和': 2018,  # 2018年が令和元年
    }
    for era, start_year in era_map.items():
        if era in date_str:
            year = int(re.findall(r'\d+', date_str)[0])
            return pd.to_datetime(date_str.replace(era, str(start_year + year - 1)).replace('年', '-').replace('月', '-').replace('日', ''), format='%Y-%m-%d')
    return pd.to_datetime(date_str)  # 元号が見つからない場合はそのまま返す

# 日付を西暦に変換
data['発生日'] = data['発生日'].apply(convert_era_to_western)

# 変換後のデータを再確認
data.head()

# 日付の元号を西暦に変換する関数を修正
def convert_era_to_western(date_str):
    era_map = {
        '令和': 2019,  # 令和元年は2019年
    }
    for era, start_year in era_map.items():
        if era in date_str:
            year_number = re.search(r'\d+', date_str)
            year = int(year_number.group()) if year_number else 1  # 元年の場合は年数が記載されていないため、1とする
            new_year = start_year + year - 1
            new_date_str = date_str.replace(era, '').replace(str(year), str(new_year))
            new_date_str = new_date_str.replace('年', '-').replace('月', '-').replace('日', '')
            return pd.to_datetime(new_date_str, errors='coerce')
    return pd.to_datetime(date_str, errors='coerce')  # エラー時はNaTを返す

# 日付を西暦に変換
data['発生日'] = data['発生日'].apply(convert_era_to_western)

# 変換後のデータを再確認
data.head()

# 飼養羽数の数値化の関数を再度定義（万単位の処理を含む）
def parse_birds(birds_str):
    birds_str = birds_str.replace('羽', '').strip()
    if '万' in birds_str:
        number, unit = birds_str.split('万')
        return float(number) * 10000  # 万単位を数値に変換
    return float(birds_str)  # 通常の数値変換

# 飼養羽数を数値化
data['飼養羽数'] = data['飼養羽数'].apply(parse_birds)

# 不要なカラムの削除
data.drop(columns=['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'], inplace=True)

# 緯度経度データの再度確認と割り当て
data['latitude'] = data['発生場所'].map(lambda x: coordinates.get(x, (None, None))[0])
data['longitude'] = data['発生場所'].map(lambda x: coordinates.get(x, (None, None))[1])

# 最終的なデータの確認
data.head()

# 飼養羽数の数値化の関数を修正（コンマの削除を追加）
def parse_birds(birds_str):
    birds_str = birds_str.replace('羽', '').replace(',', '').strip()
    if '万' in birds_str:
        number, unit = birds_str.split('万')
        return float(number) * 10000  # 万単位を数値に変換
    return float(birds_str)  # 通常の数値変換

# 飼養羽数を数値化
data['飼養羽数'] = data['飼養羽数'].apply(parse_birds)

# 最終的なデータの確認
data.head()

# 不要なカラムの削除
data.drop(columns=['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'], inplace=True)

# 緯度経度データの再度確認と割り当て
data['latitude'] = data['発生場所'].map(lambda x: coordinates.get(x, (None, None))[0])
data['longitude'] = data['発生場所'].map(lambda x: coordinates.get(x, (None, None))[1])

# データクレンジング後の最終データセットを表示
data.head()

# coordinates マップを再定義
coordinates = {
    '北海道厚真町': (42.5067, 141.9064),
    '千葉県香取市': (35.8972, 140.4996),
    '新潟県上越市': (37.1483, 138.2364),
    '島根県大田市': (35.1973, 132.4999),
    '新潟県胎内市': (37.6691, 139.0469)
}

# 緯度経度データの割り当て
data['latitude'] = data['発生場所'].map(lambda x: coordinates.get(x, (None, None))[0])
data['longitude'] = data['発生場所'].map(lambda x: coordinates.get(x, (None, None))[1])

# 最終的なデータセットを表示
data.head()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 日付が欠損しているデータを除外
data_clean = data.dropna(subset=['発生日'])

# 危険度をランダムに生成（実際のアプリケーションでは適切なデータを使用）
np.random.seed(42)
data_clean['危険度'] = np.random.rand(len(data_clean)) * 100

# 特徴量とターゲットの選定
X = data_clean[['latitude', 'longitude', '飼養羽数']]
y = data_clean['危険度']

# データセットを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストモデルの訓練
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# テストデータに対する予測の実行
y_pred = model.predict(X_test)

# 結果の表示
print("予測値:", y_pred)
print("実際の値:", y_test.tolist())

# モデルの評価（省略可）
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
