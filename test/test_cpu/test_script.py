import os
import unittest


class TestScript:
    def test_default(self):
        os.system(
            """
                cd ../.. && 
                python -m auto_round
                    --iters 2
                    --deployment_device fake
                    --output_dir ./tmp_script_test"""
        )
