import cdsapi

from remake import Remake, Rule

import mcs_prime.mcs_prime_config_util as cu

client = cdsapi.Client()

slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '20:00:00'}
rmk = Remake(config=dict(slurm=slurm_config, content_checks=False))

DATADIR = cu.PATHS['datadir']

# E.g. (generated from https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download):
"""
import cdsapi

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["geopotential"],
    "year": ["2020"],
    "month": ["01"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "pressure_level": ["500"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
"""

req_var_names = {
    'geopotential': 'geopotential',
}


class Era5DownloadGeopotential(Rule):
    rule_matrix = {'month': list(range(1, 13)), 'variable': ['geopotential']}

    rule_inputs = {}
    rule_outputs = {
        'output': str(
            DATADIR / 'ecmwf-era5/misc/2020/{month:02d}/'
            'ecmwf-era5_oper_an_pl.2024.{month:02d}.1-11.{variable}_500hPa.nc'
        )
    }

    @staticmethod
    def rule_run(inputs, outputs, month, variable):
        var_name = req_var_names[variable]

        msg = f'Download {variable} for {month:02d}'
        print(msg)
        print('=' * len(msg))

        output_path = outputs['output']
        # Create a tmp path, save download to this, then mv to actual output_path.
        # Ensures that output_path will *only* be present if download has completed.

        dataset = "reanalysis-era5-pressure-levels"
        request_dict = {
            'product_type': ['reanalysis'],
            'variable': [var_name],
            'year': ['2020'],
            'month': [f'{month:02d}'],
            'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'],
            'time': [
                '00:00',
                '01:00',
                '02:00',
                '03:00',
                '04:00',
                '05:00',
                '06:00',
                '07:00',
                '08:00',
                '09:00',
                '10:00',
                '11:00',
                '12:00',
                '13:00',
                '14:00',
                '15:00',
                '16:00',
                '17:00',
                '18:00',
                '19:00',
                '20:00',
                '21:00',
                '22:00',
                '23:00',
            ],
            'pressure_level': ['500'],
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        }
        client.retrieve(
            dataset,
            request_dict,
            output_path,
        )
