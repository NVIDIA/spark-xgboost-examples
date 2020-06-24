# Build XGBoost Scala Examples

Our examples rely on [cuDF](https://github.com/rapidsai/cudf) and [XGBoost](https://github.com/nvidia/spark-xgboost)

##### Build Process

Follow these steps to build the Scala jars (Here take CUDA 10.2 as an example):

```
git clone https://github.com/NVIDIA/spark-xgboost-examples.git
cd spark-xgboost-examples/examples/apps/scala
mvn package -Dcuda.classifier=cuda10-2
```

##### Generated Jars

The build process generates two jars:

+ *sample_xgboost_apps-0.2.2.jar* : only classes for the examples are included, so it should be submitted to spark together with other dependent jars
+ *sample_xgboost_apps-0.2.2-jar-with-dependencies.jar*: both classes for the examples and the classes from dependent jars are included

##### Build Options

Classifiers:

+ *cuda.classifier*
    + For CUDA 10.1 building, specify *cuda10-1*
    + For CUDA 10.2 building, specify *cuda10-2*
###### NOTE: CUDA 10.0 is no longer supported.
