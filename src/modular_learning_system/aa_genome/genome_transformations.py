# File: /Users/derrickbass/Public/adaptai/src/modular_learning_system/aa_genome/genome_transformations.py


class GenomeTransformations:
    def scale_genomes(self, df, scaling_factor):
        """
        Apply scaling to genomes.
        :param df: Spark DataFrame containing genomes
        :param scaling_factor: Value to scale the genomes
        :return: Scaled DataFrame
        """
        return df.withColumn('scaled_genome', df['genome'] * scaling_factor)

    def filter_genomes(self, df, threshold):
        """
        Filter genomes based on a fitness threshold.
        :param df: Spark DataFrame containing genomes
        :param threshold: Fitness threshold
        :return: Filtered DataFrame
        """
        return df.filter(df['fitness'] > threshold)
