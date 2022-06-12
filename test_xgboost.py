import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


# onehot编码模型,返回编码后的数据
def onehotmodel(data):
    # 找出数字列
    num_col = data.select_dtypes(include=[np.number])
    non_num_col = data.select_dtypes(exclude=[np.number])
    # one hot编码
    onehotnum = pd.get_dummies(non_num_col)
    # one hot编码后数据并合并
    data = pd.concat([num_col, onehotnum], axis=1)
    return data


train_data = pd.read_csv(r'E:\桌面\train.csv')
test_data = pd.read_csv(r'E:\桌面\test.csv')
test_CUST_ID = test_data["CUST_ID"]
train_data_target = train_data["bad_good"]
train_data.drop(["bad_good"], axis=1, inplace=True)

dropcols = [
    "OPEN_ORG_NUM", "IDF_TYP_CD", "GENDER", "CUST_EUP_ACCT_FLAG", "CUST_AU_ACCT_FLAG",
    "CUST_DOLLER_FLAG", "CUST_INTERNATIONAL_GOLD_FLAG", "CUST_INTERNATIONAL_COMMON_FLAG",
    "CUST_INTERNATIONAL_SIL_FLAG", "CUST_INTERNATIONAL_DIAMOND_FLAG", "CUST_GOLD_COMMON_FLAG",
    "CUST_STAD_PLATINUM_FLAG", "CUST_LUXURY_PLATINUM_FLAG", "CUST_PLATINUM_FINANCIAL_FLAG", "CUST_DIAMOND_FLAG",
    "CUST_INFINIT_FLAG", "CUST_BUSINESS_FLAG",
]
# 在原基础上，删除没有意义的列
train_data.drop(dropcols, axis=1, inplace=True)
test_data.drop(dropcols, axis=1, inplace=True)

# 删除训练集中的重复行
train_data = train_data.drop_duplicates(keep="first")
# 去除包含NaN的行
train_data.dropna(inplace=True)

# 对train、test的数据进行 Onehot 编码
train_data = onehotmodel(train_data)
test_data = onehotmodel(test_data)


x_train, x_test, y_train, y_test = train_test_split(
    train_data, train_data_target, test_size=0.2
)

XGB = XGBClassifier(nthread=4,  # 含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                    learning_rate=0.08,  # 含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                    n_estimators=50,  # 含义：总共迭代的次数，即决策树的个数
                    max_depth=5,  # 含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                    gamma=0,  # 含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                    subsample=0.9,  # 含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止过拟合
                    colsample_bytree=0.5)  # 训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止过拟合

score = cross_val_score(XGB, train_data, train_data_target, cv=5).mean()
print("在训练集上的平均5折交叉验证分数:", score)

# 通过上面构造好的模型,对训练数据集进行评估
model = XGB.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("验证数据的准确率: ", accuracy_score(y_test, y_pred))
print("验证数据的精确率: ", precision_score(y_test, y_pred))
print("验证数据的召回率: ", recall_score(y_test, y_pred))
print("验证数据的Macro-F1: ", f1_score(y_test, y_pred, average="macro"))

# 使用上述模型,对测试数据进行预测
test_pred = model.predict(test_data)
test_pred = pd.DataFrame(test_pred, columns=["bad_good"])
sub = pd.concat([test_CUST_ID, test_pred], axis=1)
sub.to_csv(r'E:\桌面\submission.csv')

