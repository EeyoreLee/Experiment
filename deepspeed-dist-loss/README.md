# 分布式训练中的Loss有什么要求

## 存在参数要怎么写？
直接 xx.cuda()即可，但看起来每张GPU上都存在独立的一个loss function


## 如何分配的loss设备？



## 是否loss需要all gather