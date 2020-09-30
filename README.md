This repo provides docs and example applications that demonstrate the RAPIDS.ai GPU-accelerated XGBoost-Spark project. It now supports Spark 3.0.0 and 3.0.1.

### Examples

- Mortgage: [Scala](/examples/apps/scala/src/main/scala/com/nvidia/spark/examples/mortgage), [Python](/examples/apps/python/com/nvidia/spark/examples/mortgage)
- Taxi: [Scala](/examples/apps/scala/src/main/scala/com/nvidia/spark/examples/taxi), [Python](/examples/apps/python/com/nvidia/spark/examples/taxi)
- Agaricus: [Scala](/examples/apps/scala/src/main/scala/com/nvidia/spark/examples/agaricus), [Python](/examples/apps/python/com/nvidia/spark/examples/agaricus)

### Getting Started Guides

Try one of the "Getting Started Guides" below. Please note that they target the Mortgage dataset as written, but with a few changes to `EXAMPLE_CLASS` and `dataPath`, they can be easily adapted to the Taxi or Agaricus datasets.

You can get a small size datasets for each example in the [datasets](/datasets) folder. These datasets are only provided for convenience. In order to test for performance, please prepare a larger dataset by following [Preparing Datasets via Notebook](/datasets/preparing_datasets.md). We also provide a larger dataset: [Morgage Dataset (1 GB uncompressed)](https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip), which is used in the guides below.

- Building applications
    - [Scala](/getting-started-guides/building-sample-apps/scala.md)
    - [Python](/getting-started-guides/building-sample-apps/python.md)
- Getting started on on-premises clusters
    - [Standalone cluster for Scala](/getting-started-guides/on-prem-cluster/standalone-scala.md)
    - [Standalone cluster for Python](/getting-started-guides/on-prem-cluster/standalone-python.md)
    - [YARN for Scala](/getting-started-guides/on-prem-cluster/yarn-scala.md)
    - [YARN for Python](/getting-started-guides/on-prem-cluster/yarn-python.md)
    - [Kubernetes](/getting-started-guides/on-prem-cluster/kubernetes.md)
- Getting started on cloud service providers
    - Amazon AWS
        - [EC2](/getting-started-guides/csp/aws/ec2.md)
    - [Databricks](/getting-started-guides/csp/databricks/databricks.md)
- Getting started for Jupyter Notebook applications
    - [Apache Toree Notebook for Scala](/getting-started-guides/notebook/toree.md)
    - [Jupyter Notebook for Python](/getting-started-guides/notebook/python-notebook.md)

These examples use default parameters for demo purposes. For a full list please see "Supported Parameters" for [Scala](/examples/app-parameters/supported_xgboost_parameters_scala.md) or [Python](/examples/app-parameters/supported_xgboost_parameters_python.md)

### XGBoost-Spark API

- [Scala API](/api-docs/scala.md)
- [Python API](/api-docs/python.md)

### Contact Us

Please see the [RAPIDS](https://rapids.ai/community.html) website for contact information.

### License

This content is licensed under the [Apache License 2.0](/LICENSE)
