import pandas as pd


# import pandas as pd

data = pd.read_csv('ccf_first_round_user_shop_behavior.csv')
shopdata=pd.read_csv('ccf_first_round_shop_info.csv')
m_shopdata = shopdata[shopdata.mall_id=='m_7800']
m_shop = list(set(m_shopdata['shop_id']))
print data[data.shop_id=='s_3724181']
print m_shopdata[m_shopdata.shop_id=='s_3724181']
for shop_id in m_shop:
    data1 = data[data.shop_id==shop_id]
    print shop_id,data1.shape[0]
# del shopdata
# del m_shopdata
# data = pd.read_csv('ccf_first_round_user_shop_behavior.csv')
# m_data = data[data.shop_id.isin(m_shop)]
# del data
# del m_shop
# m_data.to_csv ("m_1263data.csv" , index=False,header=True,encoding = "utf-8")
# print m_data

