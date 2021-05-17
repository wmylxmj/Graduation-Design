# Graduation-Design
毕业设计
#### environment
- tensorflow-gpu==1.12.0
- keras==2.2.4

### 训练单人姿态估计网络
~~~Python
from train import SPENetTrain
spe = SPENetTrain(layers=8, joints=17, lr=1e-4, pretrained_weights=None)
~~~

### 导入预训练权重进行训练
~~~Python
from train import SPENetTrain
spe = SPENetTrain(layers=8, joints=17, lr=1e-4, pretrained_weights=“weights/SPENet-8-17.h5”)
~~~

### 单人姿态估计网络预测骨架
~~~Python
from predict import SPENetPredict

# 导入训练好的权重
model = SPENet(layers=8)
model.load_weights("weights/SPENet-8-17.h5")

p = SPENetPredict(model)
p.predict_skeleton(“test.jpg”, save_folder="outputs", save_name="test")
~~~
