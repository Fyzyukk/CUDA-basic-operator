一些常见的CUDA基础算子实现

-----------------------------------------------------------------------------------------------------------------

包含Softmax,Gelu,Dropout,Reduce,Quantize，Relu，elementwis，Upsample等基础算子。

包含FP32，FP16等实现，以及一些常用的优化手段，如共享内存，向量读写，warp-level操作，以及使用nsight computer查看带宽，吞吐量等，进行性能分析。

代码笔记链接：https://telling-magnolia-4bd.notion.site/CUDA-150137289dcd80fd960ef364c9c10584?pvs=4

笔记包含算子实现代码个人理解，笔记软件为Notion ，代码软件Clion或VScode都可以。

Windows环境配置，如果是linux环境根据版本，可能会有一些头文件报错，更换相应的版本即可。

----------------------------------------------------------------------------------------------------------------------

更新了笔记的PDF版本。以及新的自己理解算子，进行ncu-proj进行性能分析

![Image text](https://github.com/Fyzyukk/CUDA-basic-operator/blob/main/images/1.png)

![Image text](https://github.com/Fyzyukk/CUDA-basic-operator/blob/main/images/2.png)

![Image text](https://github.com/Fyzyukk/CUDA-basic-operator/blob/main/images/3.png)

![Image text](https://github.com/Fyzyukk/CUDA-basic-operator/blob/main/images/4.png)
