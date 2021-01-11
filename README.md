# Ensemble-Oversampling-

Oversampling is a crucial part in today’s world as most of the data we encounter in real scenario comes with high biasness and to solve the problem of biasness we can either make changes in the loss function of the model on which we are training, which is model specific or we can use sampling methods which can be either undersampling or oversampling. Undersampling generally results in loss of information so in most of the cases oversampling is preferred. But when we look at the available oversampling methods it does the oversampling using the methods which belong to one class like cluster based oversampling or tree based or something else which results in loss in the variation. Our approach is to add variation in the data by ensembling oversampling technique which will be using multiple different classes of methods and subsetting the data after oversampling by finding the similarity with original data, to make the size of the dataset close to oversampling using one method. Apart from this the same method without subsetting gives data augmentation which will be used in fitting neural networks that require good amount of data.

How to use the code
1.	Use jupyter notebook to open code.ipynb
2.	Add your raw data in raw data folder
3.	Give the path of directory mentioned in code.ipynb

Please access the code and raw data files using below mentioned link-
https://drive.google.com/drive/folders/18hIIgjSpkMILVFWM04St4J6DdNaF5Z2M?usp=sharing
All the Code and Raw Data Files are stored inside the folder named as "Methods"
