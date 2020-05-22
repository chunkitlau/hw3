# hw3
This is a 11-category food picture classification problem.
It is a problem from ML2020spring - hw3, Hung-yi Lee, NTU.
Kaggle link: https://www.kaggle.com/c/ml2020spring-hw3
ignore hw3.py
## demo_vgg8
hw3_demo_vgg8.py:
This is a demo given by teacher or Ta.
Github link: https://github.com/Iallen520/lhy_DL_Hw/blob/master/hw3_CNN.ipynb
After running 30 epochs with Adam, the accuracy rate on the vilidatin set of hw3 data set is about 60%.
It basically doesn't work well, so you may ignore it or learn cnn by it.
ps: It may contains some mistakes, but it is basically right.
## vgg16_bn
hw3_vgg16_bn.py:
This is a bad example of transfer learning.
After running about 60 epochs with Adam, the accuracy rate on the vilidatin set of hw3 data set is ouver 80%.
The problem is this example haven't utilize pre-trained model well.
Although it use pre-trained model, it hasn't freeze the parameters of the previous layer to train the model.
## resnet50
hw3_resnet50.py
This is a acceptable example of transfer learning.
It fix the bug mentioned above.
After running about 10 epochs with parameters of pre-trained model fixed(learing rate = 0.001), the accuracy rate on the vilidatin set of hw3 data set is about 75%.
After running about 10 another epochs with most of parameters of pre-trained model fixed, the accuracy rate on the vilidatin set of hw3 data set is nearly 90%.
This technique is called fine-tune, which means unfreezing the deep layer parameters and applying small learing rate(about 0.0001).
ps: The code still has some bugs to be fix.