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

#### 4. Congrats. !!! Do your first tracking
```python
with mlflow.start_run():
  mlflow.log_param('foo', 1000)
```

> `with mlflow.start_run` is a context manager

#### 5. Use trace to track a function params and return

> Define a function and decore it with ```@mlflow.trace``` to track call

```python
@mlflow.trace
def foo(**kwargs):
  return 0.0123
```

#### 6. Store a diagram

> To store a fig or raw file, we can use ```mlflow.log_artifact```

```python
with mlflow.start_run():
  fig.write_image('pie-chart.png')
  mlflow.log_artifact('pie-chart.png')
```

#### 7. store a model

> Example : track svm model and metrics, you can also track all models

```python
with mlflow.start_run():
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
    precision = precision_score(y_true=y_test, y_pred=y_pred, average='weighted')
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    model_name = 'svm'
    mlflow.log_param("model_name", model_name)
    mlflow.log_metric(f"{model_name}_accuracy", accuracy)
    mlflow.log_metric(f"{model_name}_recall", recall)
    mlflow.log_metric(f"{model_name}_precision", precision)
    mlflow.log_metric(f"{model_name}_f1_score", f1)
    mlflow.sklearn.log_model(svm, artifact_path=f"models/{model_name}")
```

## References
[[1] Tracking](https://mlflow.org/docs/latest/tracking.html)

[[2] lab3-model-deployment](https://github.com/abdoulfataoh/lab3-model-deployment)


