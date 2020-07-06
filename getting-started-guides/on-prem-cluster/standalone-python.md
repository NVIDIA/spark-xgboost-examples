Get Started with XGBoost4J-Spark on an Apache Spark Standalone Cluster
======================================================================
This is a getting started guide to XGBoost4J-Spark on an Apache Spark3.0 Standalone Cluster. At the end of this guide, the reader will be able to run a sample Apache Spark Python application that runs on NVIDIA GPUs.

Prerequisites
-------------
* Apache Spark 3.0 Standalone Cluster (e.g.: Spark 3.0)
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 16.04/CentOS7
  * CUDA V10.1/10.2 （CUDA 10.0 is no longer supported）
  * NVIDIA driver compatible with your CUDA
  * NCCL 2.4.7
  * Python 2.7/3.4/3.5/3.6/3.7
  * NumPy

* The number of GPUs in each host dictates the number of Spark executors that can run there. Additionally, cores per Spark executor and cores per Spark task must match, such that each executor can run 1 task at any given time. For example, if each host has 4 GPUs, there should be 4 or less executors running on each host, and each executor should run at most 1 task (e.g.: a total of 4 tasks running on 4 GPUs).
* In Spark Standalone mode, the default configuration is for an executor to take up all the cores assigned to each Spark Worker. In this example, we will limit the number of cores to 1, to match our dataset. Please see https://spark.apache.org/docs/latest/spark-standalone.html for more documentation regarding Standalone configuration.
* The `SPARK_HOME` environment variable is assumed to point to the cluster's Apache Spark installation.
* Follow the steps below to enable the GPU discovery for Spark on each host, since Spark3.0 now supports GPU scheduling, and this will let Spark3 find all available GPUs on standalone cluster.
  1. Copy the spark config file from template
  ```
  cd ${SPARK_HOME}/conf/
  cp spark-defaults.conf.template spark-defaults.conf
  ```
  2. Add the following configs to the file `spark-defaults.conf`. The number in first config should NOT larger than the actual number of the GPUs on current host. This example uses 1 as below for one GPU on the host.
  ```
  spark.worker.resource.gpu.amount 1
  spark.worker.resource.gpu.discoveryScript ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh
  ```

Get Application Files, Jar and Dataset
-------------------------------
1. *samples.zip* and *main.py*: Please build the files by following the [guide](/getting-started-guides/building-sample-apps/python.md)
2. Jars: Please download the following jars:
    * [*cudf-0.14-cuda10-2.jar*](https://repo1.maven.org/maven2/ai/rapids/cudf/0.14/) for CUDA 10.2 (Here take CUDA 10.2 as an example) or [*cudf-0.14-cuda10-1.jar*](https://repo1.maven.org/maven2/ai/rapids/cudf/0.14/) for CUDA 10.1
    * [*xgboost4j_3.0-1.0.0-0.1.0.jar*](https://repo1.maven.org/maven2/com/nvidia/xgboost4j_3.0/1.0.0-0.1.0/)
    * [*xgboost4j-spark_3.0-1.0.0-0.1.0.jar*](https://repo1.maven.org/maven2/com/nvidia/xgboost4j-spark_3.0/1.0.0-0.1.0/)
    * [*rapids-4-spark_2.12-0.1.0.jar*](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/0.1.0/)
3. Dataset: https://rapidsai.github.io/demos/datasets/mortgage-data

Place dataset and other files in a local directory. In this example the dataset was unzipped in the `xgboost4j_spark/data` directory, and all other files in the `xgboost4j_spark/libs` directory.

```
[xgboost4j_spark]$ find . -type f | sort
./data/mortgage/perf/Performance_*
./data/mortgage/acq/Acquisition_*
./libs/cudf-0.14-cuda10-2.jar
./libs/main.py
./libs/rapids-4-spark_2.12-0.1.0.jar
./libs/samples.zip
./libs/xgboost4j_3.0-1.0.0-0.1.0.jar
./libs/xgboost4j-spark_3.0-1.0.0-0.1.0.jar
```

Launch a Standalone Spark Cluster
---------------------------------
0. Copy required jars to `$SPARK_HOME/jars` folder

```
cp rapids-4-spark_2.12-0.1.0.jar $SPARK_HOME/jars/
cp cudf-0.14-cuda10-2.jar $SPARK_HOME/jars/
```

1. Start the Spark Master process:

```
${SPARK_HOME}/sbin/start-master.sh
```

Note the hostname or ip address of the Master host, so that it can be given to each Worker process, in this example the Master and Worker will run on the same host.

2. Start a Spark slave process:

```
export SPARK_MASTER=spark://`hostname -f`:7077
export SPARK_CORES_PER_WORKER=1

${SPARK_HOME}/sbin/start-slave.sh ${SPARK_MASTER} -c ${SPARK_CORES_PER_WORKER}
```

Note that in this example the Master and Worker processes are both running on the same host. This is not a requirement, as long as all hosts that are used to run the Spark app have access to the dataset.

Launch Mortgage ETL Example
---------------------------
Variables required to run spark-submit command:
```
# path to xgboost4j_spark/libs
export LIBS_PATH=/home/xgboost4j_spark/lib

# Add sample.zip to py-files config
export SPARK_PY_FILES=${LIBS_PATH}/samples.zip

```

Run spark-submit
```
${SPARK_HOME}/bin/spark-submit \
    --master spark://$HOSTNAME:7077 \
    --executor-memory 32G \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.task.resource.gpu.amount=1 \
    --conf spark.plugins=com.nvidia.spark.SQLPlugin \
    --py-files ${SPARK_PY_FILES} \
    main.py \
    --mainClass='com.nvidia.spark.examples.mortgage.etl_main' \
    --format=csv \
    --dataPath="perf::/home/xgboost4j_spark/data/mortgage/perf-train/" \
    --dataPath="acq::/home/xgboost4j_spark/data/mortgage/acq-train/" \
    --dataPath="out::/home/xgboost4j_spark/data/mortgage/out/train/"

# if generating eval data, change the data path to eval as well as the corresponding perf-eval and acq-eval data
# --dataPath="perf::/home/xgboost4j_spark/data/mortgage/perf-eval"
# --dataPath="acq::/home/xgboost4j_spark/data/mortgage/acq-eval"
# --dataPath="out::/home/xgboost4j_spark/data/mortgage/out/eval/"
```

Launch GPU Mortgage Example
---------------------------
Variables required to run spark-submit command:

```
# this is the same master host we defined while launching the cluster
export SPARK_MASTER=spark://`hostname -f`:7077

# location where data was downloaded
export DATA_PATH=./xgboost4j_spark/data

# location for the required libs
export LIBS_PATH=./xgboost4j_spark/libs

# Currently the number of tasks and executors must match the number of input files.
# For this example, we will set these such that we have 1 executor, with 1 core per executor

## take up the the whole worker
export SPARK_CORES_PER_EXECUTOR=${SPARK_CORES_PER_WORKER}

## run 1 executor
export SPARK_NUM_EXECUTORS=1

## cores/executor * num_executors, which in this case is also 1, limits
## the number of cores given to the application
export TOTAL_CORES=$((SPARK_CORES_PER_EXECUTOR * SPARK_NUM_EXECUTORS))

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
 --master ${SPARK_MASTER}                                                       \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --conf spark.cores.max=${TOTAL_CORES}                                          \
 --jars ${SPARK_JARS},${JAR_RAPIDS}                                                           \
 --py-files ${SPARK_PY_FILES}                                                   \
 ${SPARK_PYTHON_ENTRYPOINT}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --dataPath=train::${DATA_PATH}/mortgage/out/train/      \
 --dataPath=trans::${DATA_PATH}/mortgage/out/eval/      \
 --format=parquet                                                                   \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8

 # Change the format to csv if your input file is CSV format.

```

In the `stdout` log on driver side, you should see timings<sup>*</sup> (in seconds), and the accuracy metric:

```
----------------------------------------------------------------------------------------------------
Training takes 14.65 seconds

----------------------------------------------------------------------------------------------------
Transformation takes 12.21 seconds

----------------------------------------------------------------------------------------------------
Accuracy is 0.9873692247091792
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
# this is the same master host we defined while launching the cluster
export SPARK_MASTER=spark://`hostname -f`:7077

# location where data was downloaded
export DATA_PATH=./xgboost4j_spark/data

# location for the required libs
export LIBS_PATH=./xgboost4j_spark/libs

# Currently the number of tasks and executors must match the number of input files.
# For this example, we will set these such that we have 1 executor, with 1 core per executor

## take up the the whole worker
export SPARK_CORES_PER_EXECUTOR=${SPARK_CORES_PER_WORKER}

## run 1 executor
export SPARK_NUM_EXECUTORS=1

## cores/executor * num_executors, which in this case is also 1, limits
## the number of cores given to the application
export TOTAL_CORES=$((SPARK_CORES_PER_EXECUTOR * SPARK_NUM_EXECUTORS))

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
 --master ${SPARK_MASTER}                                                       \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --conf spark.cores.max=${TOTAL_CORES}                                          \
 --jars ${SPARK_JARS}                                                           \
 --py-files ${SPARK_PY_FILES}                                                   \
 ${SPARK_PYTHON_ENTRYPOINT}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --dataPath=train::${DATA_PATH}/mortgage/out/train/      \
 --dataPath=trans::${DATA_PATH}/mortgage/out/eval/         \
 --format=parquet                                                               \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8

 # Change the format to csv if your input file is CSV format.
 
```

In the `stdout` log on driver side, you should see timings<sup>*</sup> (in seconds), and the accuracy metric:

```
----------------------------------------------------------------------------------------------------
Training takes 225.7 seconds

----------------------------------------------------------------------------------------------------
Transformation takes 36.26 seconds

----------------------------------------------------------------------------------------------------
Accuracy is 0.9873709530950067
```

<sup>*</sup> The timings in this Getting Started guide are only illustrative. Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

