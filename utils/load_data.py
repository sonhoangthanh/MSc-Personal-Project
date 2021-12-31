# Imports
import pandas as pd
import numpy as np


def load_data(data_path, mode):
    # Variables 
    target_population = 80e3  # population of Isle of Man
    input_population = 70e6  # population of UK

    # Load the power data
    solar_data = pd.read_csv(data_path + 'ninja_pv_25MW.csv', header=3)
    wind_data = pd.read_csv(data_path + 'ninja_wind_80MW.csv', header=3)

    # Solar and Wind power is in kW
    solar_power = solar_data['electricity']
    wind_power = wind_data['electricity']

    load_power = None

    if mode == 'train' or mode == 'tune':
        # Need to scale the UK demand to the population of Isle of Man target 
        load_power = pd.read_csv(data_path + 'gridwatch2019.csv')[' demand'] * target_population / input_population * 1000  # multiplied by 1000 because of unit conversion from MW to kW

    elif mode == 'eval':
        load_power = pd.read_csv(data_path + 'gridwatch2020.csv')[' demand'] * target_population / input_population * 1000  # multiplied by 1000 because of unit conversion from MW to kW


    # Need to generate synthetic 
    # Start with having the power generated be varying with 5% from the initial data
    # Conatenate together the dataframes
    if mode == 'train' or mode == 'tune':
        
        load_extended = load_power.append(load_power)
        solar_extended = solar_power.append(solar_power)
        wind_extended = wind_power.append(wind_power)

        # Clean data from outlyers
        load_extended = load_extended[load_extended[:] > 15e3]
        # Load is in precision of 5 min, change this to 1 hour by picking every 12th element
        load_extended = load_extended[::12]
        load_extended = load_extended.reset_index(drop=True)  # reset the index due to slicing

        # Limit the number of points the available number of load data
        solar_extended = solar_extended[:len(load_extended)]
        wind_extended = wind_extended[:len(load_extended)]

        # Add some random noise of +-10%
        random_noise_load = np.random.uniform(-0.1, 0.1, len(load_extended))
        random_noise_solar = np.random.uniform(-0.1, 0.1, len(load_extended))
        random_noise_wind = np.random.uniform(-0.1, 0.1, len(load_extended))

        load_full = load_extended + load_extended * random_noise_load
        solar_full = solar_extended + solar_extended * random_noise_solar
        wind_full = wind_extended + wind_extended * random_noise_wind

        return solar_full.to_numpy(), wind_full.to_numpy(), load_full.to_numpy()

    elif mode == 'eval':
        # Clean data from outlyers
        load_power = load_power[load_power[:] > 15e3]
        # Need to only take data from the hourly readings, not every 5 min. Need to thus take only every 12th entry
        load_adjusted = load_power[::12]
        
        # Get only the power column and the same number of data points as for load 
        load_adjusted = load_adjusted[:len(solar_power)] 
        # Reset the sliced index
        load_adjusted = load_adjusted.reset_index(drop=True)

        return solar_power, wind_power, load_adjusted

    else:
        raise Exception('Incorrect Mode!')