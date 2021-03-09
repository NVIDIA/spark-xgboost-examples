# Build XGBoost Scala Examples

The examples rely on [XGBoost](https://github.com/nvidia/spark-xgboost).

## Build

Follow these steps to build the Scala jars:

``` bash
git clone https://github.com/NVIDIA/spark-xgboost-examples.git
cd spark-xgboost-examples/examples/apps/scala
mvn package
```

## The generated Jars

Let's assume LATEST_VERSION is **0.2.2**. The build process will generate two jars as belows,

+ *sample_xgboost_apps-${LATEST_VERSION}.jar*
  
  only classes for the examples are included, so it should be submitted to spark together with other dependent jars

+ *sample_xgboost_apps-${LATEST_VERSION}-jar-with-dependencies.jar*
  
  both classes for the examples and the classes from dependent jars are included except cudf and rapids.
