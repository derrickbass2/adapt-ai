import os
import sys
import unittest

# Change the import path for aa_genome
sys.path.insert(0, os.path.abspath('../../mls'))

class TestAAGenome(unittest.TestCase):
    def test_train_AA_genome_model(self):
        pass  # Implement your test cases for train_AA_genome_model

    def test_test_AA_genome_model(self):
        pass  # Implement your test cases for test_AA_genome_model

if __name__ == "__main__":
    unittest.main()