# coding=utf-8
import numpy as np
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()  # 取得knn分类器
data = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5],
                 [98, 2]])  # <span style="font-family:Arial, Helvetica, sans-serif;">data对应着打斗次数和接吻次数</span>
labels = np.array(
    [1, 1, 1, 2, 2, 2])  # <span style="font-family:Arial, Helvetica, sans-serif;">labels则是对应Romance和Action</span>
knn.fit(data, labels)  # 导入数据进行训练'''
test = np.array([18, 90])
result = knn.predict(test.reshape(1, -1))
print(result)
