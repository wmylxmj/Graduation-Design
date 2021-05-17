# Graduation-Design
毕业设计
#### environment
- tensorflow-gpu==1.12.0
- keras==2.2.4

### 训练单人姿态估计网络
···
from train import SPENetTrain
spe = SPENetTrain(layers=8, joints=17, lr=1e-4, pretrained_weights=None)
···

