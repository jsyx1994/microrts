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
运行java游戏引擎，通过地图编辑器编辑地图
