import pytest
from importlib import resources

from cinnabar import FEMap


@pytest.fixture(scope='session')
def example_csv():
    with resources.path("cinnabar.data", "example.csv") as fn:
        yield str(fn)


@pytest.fixture(scope="session")
def fe_map(example_csv):
    """FEMap using test csv data"""
    return FEMap.from_csv(example_csv)


@pytest.fixture()
def ref_legacy():
    # a json dump of the .graph attribute created by arsenic
    return """\
{
 "CAT-13a": {
  "CAT-13m": {
   "calc_DDG": -0.95,
   "calc_dDDG": 0.13,
   "exp_DDG": 0.08000000000000007,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17g": {
   "calc_DDG": -0.02,
   "calc_dDDG": 0.1,
   "exp_DDG": -0.9000000000000004,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17i": {
   "calc_DDG": -0.76,
   "calc_dDDG": 0.11,
   "exp_DDG": -0.6300000000000008,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-13b": {
  "CAT-17g": {
   "calc_DDG": 0.36,
   "calc_dDDG": 0.11,
   "exp_DDG": -0.620000000000001,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-13c": {
  "CAT-17i": {
   "calc_DDG": 0.26,
   "calc_dDDG": 0.11,
   "exp_DDG": -0.15000000000000036,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-13d": {
  "CAT-13b": {
   "calc_DDG": 2.12,
   "calc_dDDG": 0.12,
   "exp_DDG": 1.3500000000000014,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-13f": {
   "calc_DDG": 0.13,
   "calc_dDDG": 0.13,
   "exp_DDG": 1.3800000000000008,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-13h": {
   "calc_DDG": 1.46,
   "calc_dDDG": 0.1,
   "exp_DDG": 0.8400000000000016,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-13i": {
   "calc_DDG": -0.59,
   "calc_dDDG": 0.13,
   "exp_DDG": 1.200000000000001,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17a": {
   "calc_DDG": -0.78,
   "calc_dDDG": 0.07,
   "exp_DDG": -0.2599999999999998,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17d": {
   "calc_DDG": 2.71,
   "calc_dDDG": 0.09,
   "exp_DDG": 1.0500000000000007,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17h": {
   "calc_DDG": 0.05,
   "calc_dDDG": 0.07,
   "exp_DDG": 0.14000000000000057,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-13e": {
  "CAT-17g": {
   "calc_DDG": 1.5,
   "calc_dDDG": 0.11,
   "exp_DDG": 0.21999999999999886,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17i": {
   "calc_DDG": 1.04,
   "calc_dDDG": 0.11,
   "exp_DDG": 0.48999999999999844,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-13f": {},
 "CAT-13g": {
  "CAT-17g": {
   "calc_DDG": 0.86,
   "calc_dDDG": 0.15,
   "exp_DDG": -0.6500000000000004,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17i": {
   "calc_DDG": 0.84,
   "calc_dDDG": 0.13,
   "exp_DDG": -0.3800000000000008,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-13h": {
  "CAT-17i": {
   "calc_DDG": 0.14,
   "calc_dDDG": 0.09,
   "exp_DDG": 0.15999999999999837,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-13i": {},
 "CAT-13j": {
  "CAT-4o": {
   "calc_DDG": 0.29,
   "calc_dDDG": 0.1,
   "exp_DDG": -0.6499999999999986,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-13k": {
  "CAT-4b": {
   "calc_DDG": -1.16,
   "calc_dDDG": 0.1,
   "exp_DDG": 0.07000000000000028,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-4d": {
   "calc_DDG": -0.59,
   "calc_dDDG": 0.11,
   "exp_DDG": 0.5899999999999999,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-13m": {},
 "CAT-13n": {
  "CAT-13a": {
   "calc_DDG": -1.73,
   "calc_dDDG": 0.12,
   "exp_DDG": -0.3000000000000007,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-13k": {
   "calc_DDG": -2.98,
   "calc_dDDG": 0.14,
   "exp_DDG": -1.1600000000000001,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-4i": {
   "calc_DDG": -0.23,
   "calc_dDDG": 0.12,
   "exp_DDG": 0.27999999999999936,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-13o": {
  "CAT-17h": {
   "calc_DDG": -1.98,
   "calc_dDDG": 0.1,
   "exp_DDG": -1.790000000000001,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17i": {
   "calc_DDG": -1.08,
   "calc_dDDG": 0.12,
   "exp_DDG": -0.9300000000000015,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-17a": {},
 "CAT-17b": {
  "CAT-13d": {
   "calc_DDG": 0.46,
   "calc_dDDG": 0.09,
   "exp_DDG": -0.45000000000000107,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17e": {
   "calc_DDG": 0.38,
   "calc_dDDG": 0.08,
   "exp_DDG": 0.0,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-17c": {
  "CAT-17e": {
   "calc_DDG": 0.38,
   "calc_dDDG": 0.11,
   "exp_DDG": -0.16000000000000014,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-17d": {},
 "CAT-17e": {},
 "CAT-17f": {
  "CAT-17e": {
   "calc_DDG": -0.03,
   "calc_dDDG": 0.07,
   "exp_DDG": -0.5999999999999996,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-17g": {
  "CAT-13c": {
   "calc_DDG": 0.05,
   "calc_dDDG": 0.11,
   "exp_DDG": 0.41999999999999993,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-13i": {
   "calc_DDG": -0.67,
   "calc_dDDG": 0.12,
   "exp_DDG": 0.47000000000000064,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17c": {
   "calc_DDG": -1.86,
   "calc_dDDG": 0.08,
   "exp_DDG": -0.11999999999999922,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17d": {
   "calc_DDG": 1.03,
   "calc_dDDG": 0.06,
   "exp_DDG": 0.3200000000000003,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17f": {
   "calc_DDG": -1.36,
   "calc_dDDG": 0.08,
   "exp_DDG": 0.3200000000000003,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-17h": {},
 "CAT-17i": {
  "CAT-13f": {
   "calc_DDG": -0.77,
   "calc_dDDG": 0.13,
   "exp_DDG": 0.3800000000000008,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17a": {
   "calc_DDG": -1.96,
   "calc_dDDG": 0.08,
   "exp_DDG": -1.2599999999999998,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-24": {
  "CAT-17e": {
   "calc_DDG": 1.98,
   "calc_dDDG": 0.08,
   "exp_DDG": 1.33,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-17i": {
   "calc_DDG": 2.89,
   "calc_dDDG": 0.07,
   "exp_DDG": 1.879999999999999,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-4a": {
  "CAT-13k": {
   "calc_DDG": -1.37,
   "calc_dDDG": 0.09,
   "exp_DDG": -1.7699999999999996,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-4o": {
   "calc_DDG": -0.78,
   "calc_dDDG": 0.07,
   "exp_DDG": -1.4499999999999993,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-4b": {},
 "CAT-4c": {
  "CAT-4o": {
   "calc_DDG": -1.35,
   "calc_dDDG": 0.09,
   "exp_DDG": -1.5299999999999994,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-4d": {},
 "CAT-4i": {
  "CAT-13m": {
   "calc_DDG": -3.07,
   "calc_dDDG": 0.12,
   "exp_DDG": -0.5,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-4j": {
  "CAT-4o": {
   "calc_DDG": -0.58,
   "calc_dDDG": 0.06,
   "exp_DDG": -0.35999999999999943,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-4k": {
  "CAT-4o": {
   "calc_DDG": -1.21,
   "calc_dDDG": 0.08,
   "exp_DDG": -1.5299999999999994,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-4l": {
  "CAT-13k": {
   "calc_DDG": -2.27,
   "calc_dDDG": 0.12,
   "exp_DDG": -0.35999999999999943,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-4m": {
  "CAT-13j": {
   "calc_DDG": -0.01,
   "calc_dDDG": 0.12,
   "exp_DDG": 0.41999999999999993,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-13k": {
   "calc_DDG": -1.0,
   "calc_dDDG": 0.09,
   "exp_DDG": -0.5499999999999989,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-13m": {
   "calc_DDG": -1.96,
   "calc_dDDG": 0.12,
   "exp_DDG": 0.39000000000000057,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-4c": {
   "calc_DDG": 0.78,
   "calc_dDDG": 0.1,
   "exp_DDG": 1.3000000000000007,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-4j": {
   "calc_DDG": 0.71,
   "calc_dDDG": 0.07,
   "exp_DDG": 0.13000000000000078,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-4k": {
   "calc_DDG": 1.14,
   "calc_dDDG": 0.08,
   "exp_DDG": 1.3000000000000007,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-4l": {
   "calc_DDG": 1.05,
   "calc_dDDG": 0.11,
   "exp_DDG": -0.1899999999999995,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-4n": {
   "calc_DDG": -0.59,
   "calc_dDDG": 0.07,
   "exp_DDG": 0.0600000000000005,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-4p": {
   "calc_DDG": -1.1,
   "calc_dDDG": 0.06,
   "exp_DDG": -0.9299999999999997,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-4n": {
  "CAT-13k": {
   "calc_DDG": 0.08,
   "calc_dDDG": 0.1,
   "exp_DDG": -0.6099999999999994,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-4o": {
  "CAT-4b": {
   "calc_DDG": -1.21,
   "calc_dDDG": 0.11,
   "exp_DDG": -0.25,
   "exp_dDDG": 0.14142135623730953
  },
  "CAT-4d": {
   "calc_DDG": -1.66,
   "calc_dDDG": 0.11,
   "exp_DDG": 0.2699999999999996,
   "exp_dDDG": 0.14142135623730953
  }
 },
 "CAT-4p": {
  "CAT-13k": {
   "calc_DDG": 0.59,
   "calc_dDDG": 0.08,
   "exp_DDG": 0.3800000000000008,
   "exp_dDDG": 0.14142135623730953
  }
 }
}"""