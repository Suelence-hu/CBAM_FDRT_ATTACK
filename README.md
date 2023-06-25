# CBAM_FDRT_ATTACK
1、环境
python 3.6
pytorch 1.10.2
torchvision 0.11.3  

	2、five_common_attacks.py 是普通的五种攻击方法，作为基准，模型为Resnet50，当然也可通过models切换模型  
	3、CBAM_ResNet50.py在模型中加入了注意力机制  
	4、FDRT_attack.py是加入了变换后的对抗样本算法  
	5、black_attacks.py用于测试模型的迁移性能，可以看到其中用到了两个模型  
	6、torchattacks文件夹中的attack文件夹包含了10种攻击的具体算法   
7、其他内容在代码中均有注释
