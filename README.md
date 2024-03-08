## MUTAN-for-VQAv2

This project is to reproduce the resulting code from the paper MUTAN: Multimodal Tucker Fusion for Visual Question Answering, and I will teach you in detail how to run through this code, including the processing of the dataset.
## 1.项目结构
||==> AttenMUTANmodels #带有注意力的MUTAN“模型文件”

  	|==> image_feature.py #视觉特征提取器

  	|==> bert_encoder.py #文本特征提取器

 	 |==> fusion.py #Tucker Fusioon 脚本

 	 |==> My_Att_MUTANmodel.py #模型文件

||==> baseline_models #不带注意力的基线模型文件

  	|==> image_feature.py #视觉特征提取器

  	|==> bert_encoder.py #文本特征提取器

  	|==> fusion.py #Tucker Fusioon 脚本

 	|==> MUTANmodel.py #模型文件

||==> Evaluate_Vqav2 #评估模型的文件

  	|==> PythonEvaluationTools #模型评估工具

​			|==> vqaEvaluation #里面存放了一个评估模型所用的类vqaEval.py

 	 |==> PythonHelperTools #这个也是官方给的文件

​			|==> vqaTools #里面存放了一个vqa.py文件

​	  |==> QuestionTypes #问题类型文件夹

 	 |==> Results #存放模型输出结果的文件

  	|==> evaluate.py #评估模型的主要文件