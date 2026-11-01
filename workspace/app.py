import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title='Fahasa Trending', page_icon='📚', layout='wide')
st.title('📚 Fahasa Trending: dự đoán giá, Bestseller và cụm sản phẩm')

COVER = ['Bìa Mềm', 'Bìa Cứng', 'Bộ Hộp', 'Board Book', 'Mixed Media Product', 'Không áp dụng']
LVL1 = ['SÁCH TIẾNG VIỆT', 'FOREIGN BOOKS', 'VĂN PHÒNG PHẨM - DỤNG ...', 'ĐỒ CHƠI', 'LƯU NIỆM']
COT_CLUSTER = ['log_Price', 'log_Qty', 'log_Pages', 'log_PricePerPage', 'Is_set', 'Has_gift']
COT_REG = ['log_Qty', 'log_Pages', 'Is_set', 'Has_gift', 'Pages_missing']
COT_CLF = ['log_Price', 'log_Pages', 'Is_set', 'Has_gift', 'Pages_missing']

df = pd.read_csv('Data/fahasa_cleaned.csv')
for cot in ['Is_set', 'Has_gift', 'Pages_missing']:
    df[cot] = df[cot].astype(int)

def dummies(d):
    out = pd.DataFrame(index=d.index)
    for c in COVER:
        out['Cover_' + c] = (d['CoverType'] == c).astype(int)
    for g in LVL1:
        out['Lvl1_' + g] = (d['Lvl1'] == g).astype(int)
    return out

@st.cache_resource
def train():
    X1 = pd.concat([df[COT_CLUSTER], dummies(df)], axis=1)
    sc1 = StandardScaler().fit(X1)
    pca = PCA(n_components=0.90, svd_solver='full', random_state=42).fit(sc1.transform(X1))
    Xp = pca.transform(sc1.transform(X1))
    km = KMeans(n_clusters=4, n_init=10, random_state=3000).fit(Xp)
    X2 = pd.concat([df[COT_REG], dummies(df)], axis=1)
    reg = LinearRegression().fit(X2, df['log_Price'])
    X3 = pd.concat([df[COT_CLF], dummies(df)], axis=1)
    sc3 = StandardScaler().fit(X3)
    logit = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42).fit(sc3.transform(X3), df['Bestseller'])
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight='balanced', random_state=42).fit(X3, df['Bestseller'])
    return sc1, pca, Xp, km, reg, sc3, logit, tree

sc1, pca, Xp, km, reg, sc3, logit, tree = train()
df['Cluster_Label'] = km.labels_

tab1, tab2 = st.tabs(['🔮 Dự đoán', '📊 PCA / Cụm'])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Thông tin sản phẩm')
        lvl1 = st.selectbox('Nhóm hàng', LVL1)
        cover = st.selectbox('Loại bìa', COVER)
        pages = st.slider('Số trang', 10, 2000, 250, 10)
        qty = st.number_input('Số lượng đã bán (cho dự đoán giá và phân cụm)', 0, 100000, 1000, 100)
        price = st.number_input('Giá bán dự kiến, VND (cho dự đoán Bestseller và phân cụm)', 1000, 5000000, 120000, 5000)
        is_set = st.checkbox('Là bộ / combo / boxset')
        gift = st.checkbox('Có quà tặng kèm')
        model_name = st.radio('Mô hình phân loại', ['Logistic Regression', 'Decision Tree'], horizontal=True)

    one = pd.DataFrame([{'Lvl1': lvl1, 'CoverType': cover, 'Is_set': int(is_set), 'Has_gift': int(gift),
                         'Pages_missing': 0, 'log_Pages': np.log10(pages + 1), 'log_Qty': np.log10(qty + 1),
                         'log_Price': np.log10(price), 'log_PricePerPage': np.log10(price / (pages + 1))}])

    with c2:
        st.subheader('Kết quả')
        gia = 10 ** reg.predict(pd.concat([one[COT_REG], dummies(one)], axis=1))[0]
        st.metric('Giá gợi ý (Linear Regression)', f'{gia:,.0f} VND')
        st.caption('Mô hình chỉ giải thích khoảng 26% biến thiên giá (R² ≈ 0,26): dùng làm mốc tham khảo.')

        Xb = pd.concat([one[COT_CLF], dummies(one)], axis=1)
        if model_name == 'Logistic Regression':
            proba = logit.predict_proba(sc3.transform(Xb))[0, 1]
        else:
            proba = tree.predict_proba(Xb)[0, 1]
        st.metric('Xác suất Bestseller', f'{proba:.0%}')
        st.progress(float(proba))
        if proba >= 0.5:
            st.success('Khả năng cao là Bestseller')
        else:
            st.info('Khả năng là sản phẩm thường')

        Xc = pd.concat([one[COT_CLUSTER], dummies(one)], axis=1)
        diem = pca.transform(sc1.transform(Xc))
        cum = int(km.predict(diem)[0])
        st.metric('Cụm sản phẩm', f'Cụm {cum}')
        st.dataframe(df[df['Cluster_Label'] == cum][['Price', 'QuantitySold', 'Pages', 'Is_set', 'Has_gift', 'Bestseller']]
                     .median().round(2).rename('Median của cụm'))

with tab2:
    ev = pca.explained_variance_ratio_
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(range(1, len(ev) + 1), np.cumsum(ev), marker='o')
        ax.axhline(0.9, ls='--', color='gray')
        ax.set_xlabel('Số PC')
        ax.set_ylabel('Phương sai tích luỹ')
        ax.set_title('PCA: explained variance')
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        sns.scatterplot(x=Xp[:, 0], y=Xp[:, 1], hue=df['Cluster_Label'], palette='tab10', s=18, ax=ax)
        ax.scatter(diem[0, 0], diem[0, 1], marker='*', s=300, color='red', label='Sản phẩm của bạn')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('K-Means (K=4)')
        ax.legend()
        st.pyplot(fig)
    st.subheader('Hồ sơ từng cụm')
    st.dataframe(df.groupby('Cluster_Label').agg(so_sp=('Price', 'size'), gia_median=('Price', 'median'),
                 ban_median=('QuantitySold', 'median'), ti_le_bestseller=('Bestseller', 'mean')).round(2))
