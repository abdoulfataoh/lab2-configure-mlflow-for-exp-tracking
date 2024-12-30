### lab2-configure-mlflow-for-exp-tracking
In this lab, you will configure MLflow to track and log machine learning experiments, including parameters, metrics, and models from an existing machine learning project.

#### 0. Take a Tour of the Project: heart-attack-predict (using 8+ models)

> ```heart-attack-predict (using + 8 models)``` is a machine learning project using 8 models to experiment with premature detection of heart attacks.
> 
> We will use this project in this lab. Please download the project to your laptop from one of these sources or use it online if you prefer (recommanded)

- [Kaggle](https://www.kaggle.com/code/abdoulfataoh/heart-attack-predict-using-8-models)
- [Google Colab](https://colab.research.google.com/drive/1oA93A3AzjcdS7AbooxMcA0lKYfxyRP2x)
  

> The following instructions are to be done on the heart attack project notebook



#### 1. Install mlflow

```bash
!pip install mlflow
```

#### 2. Connect notebook to mlflow server

```python
import mlflow

mlflow.set_tracking_uri(uri="http://<host>:<port>")
```
> Replace ```http://<host>:<port>``` to your mlflow server uri; Here ```http://localhost:8000``` [(Lab1 config)](https://github.com/abdoulfataoh/lab1-install-mlflow/edit/main/README.md)

#### 3. Set Expiriment name

```python
mlflow.set_experiment('project-or-expriment-name')
```

> For this lab, I suggest you to give this as experiment name ```heart-attack-{your-name}```
