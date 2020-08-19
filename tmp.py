from pyspark import SparkContext

sc = SparkContext()

data = sc.textFile(path).map(lambda)
