Get Started with XGBoost4J-Spark with Jupyter Notebook
===================================================================
This is a getting started guide to XGBoost4J-Spark using an [Jupyter notebook](https://jupyter.org/). At the end of this guide, the reader will be able to run a sample notebook that runs on NVIDIA GPUs.

Before you begin, please ensure that you have setup a [Spark Standalone Cluster](/getting-started-guides/on-prem-cluster/standalone-python.md).

It is assumed that the `SPARK_MASTER` and `SPARK_HOME` environment variables are defined and point to the master spark URL (e.g. `spark://localhost:7077`), and the home directory for Apache Spark respectively.

1. Make sure you have [Jupyter notebook installed](https://jupyter.org/install.html). If you install it with conda, please makes sure your Python version is consistent.

2. Make sure you have below jars.

``` bash
export CUDF_JAR=cudf-0.18-cuda10.1.jar
export RAPIDS_JAR=rapids-4-spark_2.12-0.4.0.jar
export SAMPLE_JAR=sample_xgboost_apps-0.2.2-jar-with-dependencies.jar
export XGBOOST4J_JAR=xgboost4j_3.0-1.3.0-0.1.0.jar
export XGBOOST4J_SPARK_JAR=xgboost4j-spark_3.0-1.3.0-0.1.0.jar
```

- *samples.zip* and *main.py*: build the files by following the [guide](/getting-started-guides/building-sample-apps/python.md)
- Jars: download the following jars:
    * [*cudf-latest.jar*](https://repo1.maven.org/maven2/ai/rapids/cudf/0.18/) 
    * [*xgboost4j-latest.jar*](https://repo1.maven.org/maven2/com/nvidia/xgboost4j_3.0/1.3.0-0.1.0/)
    * [*xgboost4j-spark-latest.jar*](https://repo1.maven.org/maven2/com/nvidia/xgboost4j-spark_3.0/1.3.0-0.1.0/)
    * [*rapids-latest.jar*](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/0.4.0/)

3. Go to the project root directory and launch the notebook:
  ``` bash
  PYSPARK_DRIVER_PYTHON=jupyter       \
  PYSPARK_DRIVER_PYTHON_OPTS=notebook \
  pyspark                             \
  --master ${SPARK_MASTER}            \
  --jars ${CUDF_JAR},${RAPIDS_JAR},${XGBOOST4J_JAR},${XGBOOST4J_SPARK_JAR}\
  --py-files ${XGBOOST4J_SPARK_JAR},samples.zip      \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.rapids.memory.gpu.pooling.enabled=false \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=1 \
  --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
  --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh
  ```

Then you start your notebook and open [`mortgage-gpu.ipynb`](/examples/notebooks/python/mortgage-gpu.ipynb) to explore.
