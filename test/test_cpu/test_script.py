import os
import sys
import unittest

sys.path.insert(0, "../..")


class TestScript(unittest.TestCase):
    def test_default(self):
        os.system(
            """
                cd ../.. && 
                python -m auto_round
                    --iters 2
                    --deployment_device fake
                    --output_dir ./tmp_script_test"""
        )


if __name__ == "__main__":
    unittest.main()
