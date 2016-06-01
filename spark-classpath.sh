SPARK_HOME=/usr/local/spark/spark-1.6.1-bin-without-hadoop
HADOOP_HOME=/usr/local/hadoop/hadoop-2.7.1
export SPARK_CLASSPATH=\
$SPARK_HOME/lib/spark-assembly-1.6.1-hadoop2.2.0.jar:\
$HADOOP_HOME/share/hadoop/common/\*:\
$HADOOP_HOME/share/hadoop/common/lib/\*:\
$HADOOP_HOME/share/hadoop/hdfs/\*:\
$HADOOP_HOME/share/hadoop/hdfs/lib/\*:\
$HADOOP_HOME/share/hadoop/mapreduce/\*:\
$HADOOP_HOME/share/hadoop/mapreduce/lib/\*:\
$HADOOP_HOME/share/hadoop/tools/lib/\*:\
$HADOOP_HOME/share/hadoop/yarn/\*:\
$HADOOP_HOME/share/hadoop/yarn/lib/\*:\
$HADOOP_HOME/share/hadoop/yarn/test/\*
export CLASSPATH=$SPARK_CLASSPATH
