# microrts
## 安装

安装 Python3.7(虚拟环境可自选)
- `pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`


## 重要文件目录
- test.py 单线程训练的入口，具体传参数见main函数注解 
- test_paralell.py 多线程训练入口，比上面的多加入了“线程数”和“是否为“联盟学习”，联盟学习的参数为一连串AI名，使用“,”分隔
- evaluate.py 用于测试训练结果的程序，计算对句后的胜负平百分率

- microrts/algo ，paper算法相关实现
- microrts/algo/agents， 智能体做决策的统一接口（数据->决策）
- microrts/algo/model， 模型的定义相关


- microrts/rts_wrapper ，java相关内容的对接（包括地图、使用gym对java环境的包装等）
- microrts/rts_wrapper/player，gym机制的底层实现
- microrts/settings.py， 项目目录设置等
- microrts/rts_wrapper/microrts-java.jar 由java引擎生成的jar包


## java程序的使用（microrts-java仓库）
- 在每次修改java程序后（比如python与java的通信机制），需要编译并生成对应的jar文件，替换- microrts/rts_wrapper/microrts-java.jar

## 自定义地图
运行java游戏引擎，通过地图编辑器编辑地图,保存文件到 microrts/rts_wrapper/maps，然后在`microrts/rts_wrapper/__init__.py`中添加相应的项
![image](https://user-images.githubusercontent.com/18167093/94994963-91f32e00-05cd-11eb-827a-46e49ae59767.png)

## 修改算法
位于microrts/algo均可根据效果自行调整

## 定义元数据类型（包括java端与python端的）
包含不同智能体网络的输出动作类型与其他元数据类型（例如通信的message等）
- 修改microrts/rts_wrapper/envs/datatypes.py

## 修改多线程实现
microrts/rts_wrapper/envs/multi_envs.py，可参考openai的实现

## 修改通信机制
需要同时修改java文件中gym部分<img width="817" alt="Screen Shot 2020-10-03 at 11 17 42 PM" src="https://user-images.githubusercontent.com/18167093/94995127-bc91b680-05ce-11eb-8eb6-bd8fa41c8ad7.png">
与microrts/rts_wrapper/player模块，其他相关部分可能也会涉及，记不清了

## paper
https://link.springer.com/chapter/10.1007/978-981-15-7670-6_36
