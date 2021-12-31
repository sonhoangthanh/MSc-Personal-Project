import pytest
from utils import load_data

def test_load_data():
    datapath = './data/'

    solar, wind, load = load_data(datapath, mode='train')