Get Started with XGBoost4J-Spark on Apache Hadoop YARN
======================================================
This is a getting started guide to XGBoost4J-Spark on Apache Hadoop YARN supporting GPU scheduling. At the end of this guide, the reader will be able to run a sample Apache Spark Python application that runs on NVIDIA GPUs.

Prerequisites
-------------
* Apache Spark 3.0+ running on YARN supporting GPU scheduling. (e.g.: Spark 3.0, Hadoop-Yarn 3.1.0)
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 16.04/CentOS7
  * CUDA V10.1/10.2  （CUDA 10.0 is no longer supported）
  * NVIDIA driver compatible with your CUDA
  * NCCL 2.4.7
  * Python 2.7/3.4/3.5/3.6/3.7
  * NumPy

* The number of GPUs per NodeManager dictates the number of Spark executors that can run in that NodeManager. Additionally, cores per Spark executor and cores per Spark task must match, such that each executor can run 1 task at any given time. For example: if each NodeManager has 4 GPUs, there should be 4 or less executors running on each NodeManager, and each executor should run 1 task (e.g.: A total of 4 tasks running on 4 GPUs). In order to achieve this, you may need to adjust `spark.task.cpus` and `spark.executor.cores` to match (both set to 1 by default). Additionally, we recommend adjusting `executor-memory` to divide host memory evenly amongst the number of GPUs in each NodeManager, such that Spark will schedule as many executors as there are GPUs in each NodeManager.
* The `SPARK_HOME` environment variable is assumed to point to the cluster's Apache Spark installation.
* Enable GPU scheduling and isolation in Hadoop Yarn on each host. Please refe to [here](https://hadoop.apache.org/docs/r3.1.0/hadoop-yarn/hadoop-yarn-site/UsingGpus.html) for more details.

Get Application Files, Jar and Dataset
-------------------------------
1. *samples.zip* and *main.py*: Please build the files by following the [guide](/getting-started-guides/building-sample-apps/python.md)
2. Jars: Please download the following jars:
    * [*cudf-0.14-cuda10-2.jar*](https://repo1.maven.org/maven2/ai/rapids/cudf/0.14/) for CUDA 10.2 (Here take CUDA 10.2 as an example) or [*cudf-0.14-cuda10-1.jar*](https://repo1.maven.org/maven2/ai/rapids/cudf/0.14/) for CUDA 10.1
    * [*xgboost4j_3.0-1.0.0-0.1.0.jar*](https://repo1.maven.org/maven2/com/nvidia/xgboost4j_3.0/1.0.0-0.1.0/)
    * [*xgboost4j-spark_3.0-1.0.0-0.1.0.jar*](https://repo1.maven.org/maven2/com/nvidia/xgboost4j-spark_3.0/1.0.0-0.1.0/)
    * [*rapids-4-spark_2.12-0.1.0.jar*](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/0.1.0/)
3. Dataset: https://rapidsai.github.io/demos/datasets/mortgage-data

Place dataset and other files in a local directory. In this example the dataset was unzipped in the `xgboost4j_spark_python/data` directory, and all other files in the `xgboost4j_spark_python/libs` directory.

```
[xgboost4j_spark_python]$ find . -type f | sort
./data/mortgage/perf/Performance_*
./data/mortgage/acq/Acquisition_*
./libs/cudf-0.14-cuda10-2.jar
./libs/main.py
./libs/rapids-4-spark_2.12-0.1.0.jar
./libs/samples.zip
./libs/xgboost4j_3.0-1.0.0-0.1.0.jar
./libs/xgboost4j-spark_3.0-1.0.0-0.1.0.jar
```

Create a directory in HDFS, and copy:

```
[xgboost4j_spark_python]$ hadoop fs -mkdir /tmp/xgboost4j_spark_python
[xgboost4j_spark_python]$ hadoop fs -copyFromLocal * /tmp/xgboost4j_spark_python
```

Verify that the jar and dataset are in HDFS:

```
[xgboost4j_spark_python]$ hadoop fs -find /tmp/xgboost4j_spark_python | grep "\." | sort
/tmp/xgboost4j_spark_python/data/mortgage/perf/Performance_*
/tmp/xgboost4j_spark_python/data/mortgage/acq/Acquisition_*
/tmp/xgboost4j_spark_python/libs/cudf-0.14-cuda10-2.jar
/tmp/xgboost4j_spark_python/libs/main.py
/tmp/xgboost4j_spark_python/libs/rapids-4-spark_2.12-0.1.0.jar
/tmp/xgboost4j_spark_python/libs/samples.zip
/tmp/xgboost4j_spark_python/libs/xgboost4j_3.0-1.0.0-0.1.0.jar
/tmp/xgboost4j_spark_python/libs/xgboost4j-spark_3.0-1.0.0-0.1.0.jar
```

Launch Mortgage ETL Example
---------------------------
Variables required to run spark-submit command:
```
# location where data was downloaded
export DATA_PATH=hdfs:/tmp/xgboost4j_spark_python/data

# path to xgboost4j_spark/libs
export LIBS_PATH=/home/xgboost4j_spark/lib

# additional jars for XGBoost4J example
export SPARK_JARS=${LIBS_PATH}/cudf-0.14-cuda10-2.jar

# Rapids plugin jar, working as the sql plugin on Spark3.0
export JAR_RAPIDS=${LIBS_PATH}/rapids-4-spark_2.12-0.1.0.jar

```

Run spark-submit
```
${SPARK_HOME}/bin/spark-submit \
    --master yarn 
    --deploy-mode cluster
    --jars ${SPARK_JARS},${JAR_RAPIDS}\
    main.py \
    --mainClass='com.nvidia.spark.examples.mortgage.etl_main' \
    --format=csv \
    --dataPath="perf::${DATA_PATH}/mortgage/data/mortgage/perf/" \
    --dataPath="acq::${DATA_PATH}/mortgage/data/mortgage/acq/" \
    --dataPath="out::${DATA_PATH}/mortgage/data/mortgage/out/train/"

# if generate eval data, change the data path to eval
# --dataPath="out::${DATA_PATH}/mortgage/data/mortgage/out/eval/
```

Launch GPU Mortgage Example
---------------------------
Variables required to run spark-submit command:

```
# location where data was downloaded
export DATA_PATH=hdfs:/tmp/xgboost4j_spark_python/data

# location for the required libraries
export LIBS_PATH=hdfs:/tmp/xgboost4j_spark_python/libs

# spark deploy mode (see Apache Spark documentation for more information)
export SPARK_DEPLOY_MODE=cluster

# run a single executor for this example to limit the number of spark tasks and
# partitions to 1 as currently this number must match the number of input files
export SPARK_NUM_EXECUTORS=1

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# python entrypoint
export SPARK_PYTHON_ENTRYPOINT=${LIBS_PATH}/main.py

# example class to use
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.gpu_main

# additional jars for XGBoost4J example
export SPARK_JARS=${LIBS_PATH}/cudf-0.14-cuda10-2.jar,${LIBS_PATH}/xgboost4j_3.0-1.0.0-0.1.0.jar,${LIBS_PATH}/xgboost4j-spark_3.0-1.0.0-0.1.0.jar

# Rapids plugin jar, working as the sql plugin on Spark3.0
export JAR_RAPIDS=${LIBS_PATH}/rapids-4-spark_2.12-0.1.0.jar

# additional Python files for XGBoost4J example
export SPARK_PY_FILES=${LIBS_PATH}/xgboost4j-spark_3.0-1.0.0-0.1.0.jar,${LIBS_PATH}/samples.zip

# tree construction algorithm
export TREE_METHOD=gpu_hist
```

Run spark-submit:

```
${SPARK_HOME}/bin/spark-submit                                                  \
 --conf spark.plugins=com.nvidia.spark.SQLPlugin                       \
 --conf spark.rapids.memory.gpu.pooling.enabled=false                     \
 --conf spark.executor.resource.gpu.amount=1                           \
 --conf spark.task.resource.gpu.amount=1                              \
 --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh        \
 --files ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh            \
 --master yarn                                                                  \
 --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
 --num-executors ${SPARK_NUM_EXECUTORS}                                         \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --jars ${SPARK_JARS},${JAR_RAPIDS}                                                      \
 --py-files ${SPARK_PY_FILES}                                                   \
 ${SPARK_PYTHON_ENTRYPOINT}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --dataPath=train::${DATA_PATH}/mortgage/out/train/      \
 --dataPath=trans::${DATA_PATH}/mortgage/out/eval/        \
 --format=parquet                                                                   \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8

# Change the format to csv if your input file is CSV format.
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the accuracy metric:

```
----------------------------------------------------------------------------------------------------
Training takes 10.75 seconds

----------------------------------------------------------------------------------------------------
Transformation takes 4.38 seconds

----------------------------------------------------------------------------------------------------
Accuracy is 0.997544753891
```

Launch CPU Mortgage Example
---------------------------
If you are running this example after running the GPU example above, please set these variables, to set both training and testing to run on the CPU exclusively:

```
# example class to use
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.cpu_main

# tree construction algorithm
export TREE_METHOD=hist
```

This is the full variable listing, if you are running the CPU example from scratch:

```
# location where data was downloaded
export DATA_PATH=hdfs:/tmp/xgboost4j_spark_python/data

# location for the required libraries
export LIBS_PATH=hdfs:/tmp/xgboost4j_spark_python/libs

# spark deploy mode (see Apache Spark documentation for more information)
export SPARK_DEPLOY_MODE=cluster

# run a single executor for this example to limit the number of spark tasks and
# partitions to 1 as currently this number must match the number of input files
export SPARK_NUM_EXECUTORS=1

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# python entrypoint
export SPARK_PYTHON_ENTRYPOINT=${LIBS_PATH}/main.py

# example class to use
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.cpu_main

# additional jars for XGBoost4J example
export SPARK_JARS=${LIBS_PATH}/cudf-0.14-cuda10-2.jar,${LIBS_PATH}/xgboost4j_3.0-1.0.0-0.1.0.jar,${LIBS_PATH}/xgboost4j-spark_3.0-1.0.0-0.1.0.jar

# additional Python files for XGBoost4J example
export SPARK_PY_FILES=${LIBS_PATH}/xgboost4j-spark_3.0-1.0.0-0.1.0.jar,${LIBS_PATH}/samples.zip

# tree construction algorithm
export TREE_METHOD=hist
```

This is the same command as for the GPU example, repeated for convenience:

```
${SPARK_HOME}/bin/spark-submit                                                  \
 --master yarn                                                                  \
 --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
 --num-executors ${SPARK_NUM_EXECUTORS}                                         \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --jars ${SPARK_JARS}                                                           \
 --py-files ${SPARK_PY_FILES}                                                   \
 ${SPARK_PYTHON_ENTRYPOINT}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --dataPath=train::${DATA_PATH}/mortgage/out/train/       \
 --dataPath=trans::${DATA_PATH}/mortgage/out/eval/         \
 --format=parquet                                                               \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the accuracy metric:

```
----------------------------------------------------------------------------------------------------
Training takes 10.76 seconds

----------------------------------------------------------------------------------------------------
Transformation takes 1.25 seconds

----------------------------------------------------------------------------------------------------
Accuracy is 0.998526852335
```

<sup>*</sup> The timings in this Getting Started guide are only illustrative. Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

