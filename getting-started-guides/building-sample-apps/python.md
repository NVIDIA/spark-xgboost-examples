# Build XGBoost Python Examples

## Build

Follow these steps to package the Python zip file:

``` bash
git clone https://github.com/NVIDIA/spark-xgboost-examples.git
cd spark-xgboost-examples/examples/apps/python
zip -r samples.zip com
```

## Files Required by PySpark

Two files are required by PySpark:

+ *samples.zip*
  
  the package including all example code

+ *main.py*
  
  entrypoint for PySpark, you may just copy it from folder *spark-xgboost-examples/examples/apps/python*
