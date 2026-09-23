import os
import unittest

from dysts.flows import Lorenz

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(WORKING_DIR, "tests", "test_data")


class TestModels(unittest.TestCase):
    """
    Tests integration
    """

    def test_trajectory(self):
        """
        Test generating a trajectory
        """
        model = Lorenz()
        sol = model.make_trajectory(100)
        assert sol is not None, "Generated trajectory is None"
        assert sol.shape == (100, 3), "Generated time series has the wrong shape"  # type: ignore


if __name__ == "__main__":
    unittest.main()
