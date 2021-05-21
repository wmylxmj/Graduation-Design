# Graduation-Design
毕业设计
#### 依赖库
- tensorflow-gpu==1.12.0
- keras==2.2.4
- tqdm==4.43.0
- opencv-python==4.2.0.32

#### 设置config.cfg文件来修改数据集路径

### 训练单人姿态估计网络
~~~Python
from train import SPENetTrain

spe = SPENetTrain(layers=8, joints=17, lr=1e-4, pretrained_weights=None)
spe.train(batch_size=2)
~~~

### 导入预训练权重进行训练
~~~Python
from train import SPENetTrain

spe = SPENetTrain(layers=8, joints=17, lr=1e-4, pretrained_weights=“weights/SPENet-8-17.h5”)
spe.train(batch_size=2)
~~~

### 单人姿态估计网络预测骨架
~~~Python
from predict import SPENetPredict
from model import SPENet

# 导入训练好的权重
model = SPENet(layers=8)
model.load_weights("weights/SPENet-8-17.h5")

p = SPENetPredict(model)
p.predict_skeleton(“test.jpg”, save_folder="outputs", save_name="test")
~~~

### 从视频帧中提取姿态向量并保存为json文件

~~~Python
from model import SPENet
from utils import TCPDataLoader

# 导入训练好的权重
model = SPENet(layers=8)
model.load_weights("weights/SPENet-8-17.h5")

t = TCPDataLoader()
t.prepare(model)
~~~

### 可视化json文件的输出结果

~~~Python
from utils import TCPDataLoader

t = TCPDataLoader()
# visualize输入：json路径 对应视频帧文件夹 第几帧
t.visualize()
~~~

### 训练热舒适姿态网络

~~~Python
from train import TCPNetTrain

t = TCPNetTrain(lr=2e-5)
t.train(batch_size=64, pretrained_weights=None)
~~~

### 摄像头预测

#### 运行camera.py

