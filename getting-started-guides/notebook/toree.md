Get Started with XGBoost4J-Spark with Apache Toree Jupyter Notebook
===================================================================
This is a getting started guide to XGBoost4J-Spark using an [Apache Toree](https://toree.apache.org/) Jupyter notebook. At the end of this guide, the reader will be able to run a sample notebook that runs on NVIDIA GPUs.

Before you begin, please ensure that you have setup a [Spark Standalone Cluster](/getting-started-guides/on-prem-cluster/standalone-scala.md).

It is assumed that the `SPARK_MASTER` and `SPARK_HOME` environment variables are defined and point to the master spark URL (e.g. `spark://localhost:7077`), and the home directory for Apache Spark respectively.

1. Make sure you have jupyter notebook installed first.
2. Build the 'toree' locally to support scala 2.12, and install it.
  ``` bash
  # Clone the source code
  git clone --recursive git@github.com:firestarman/incubator-toree.git -b for-scala-2.12

  # Build the Toree pip package. You can change "BASE_VERSION" to any version since it is used locally only.
  cd incubator-toree
  env BASE_VERSION=0.5.0 make pip-release

  # Install Toree
  pip install dist/toree-pip/toree-0.5.0.tar.gz
  ```

3. Install a new kernel configured for our example and with gpu enabled:

Make sure you have below jars.

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
    * 

``` bash
jupyter toree install                                \
--spark_home=${SPARK_HOME}                             \
--user                                          \
--toree_opts='--nosparkcontext'                         \
--kernel_name="XGBoost4j-Spark"                         \
--spark_opts='--master ${SPARK_MASTER} \
  --jars ${CUDF_JAR},${RAPIDS_JAR},${SAMPLE_JAR}       \
  --conf spark.sql.extensions=com.nvidia.spark.rapids.SQLExecPlugin \
  --conf spark.rapids.memory.gpu.pooling.enabled=false \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=1 \
  --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
  --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh'
```

4. Launch the notebook:
  ``` bash
  jupyter notebook
  ```

Then you start your notebook and open [`mortgage-gpu.ipynb`](/examples/notebooks/scala/mortgage-gpu.ipynb) to explore.

Please ensure that the *XGBoost4j-Spark* kernel is running.
