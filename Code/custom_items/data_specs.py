import os

input_specs_dataset = {
    'widar': {
        'dfs': (128, 2048, 6),
        'gaf': (512, 512, 6),
    },
    'signfi': {
        'dfs': (128, 256, 1),
        'gaf': (256, 256, 1)
    }
}

output_specs_dataset = {
    'widar': 6,
    'signfi': {
        'environment': 276,
        'user': 125
    }
}

# widar: 6, 1, 5, 5, user, env, position, orientation
# signfi: 5, 2 user, env
domain_size_datasets = {
    'widar': 150,
    'signfi': 10
}

domain_order_dataset = {
    'widar': {
        "struct": (6, 1, 5, 5),
        "user": 0,
        "position": 2,
        "orientation": 3
    },
    'signfi': {
        "struct": (2, 5, 1, 1),
        "environment": 0,
        "user": 1,
    },
}

SYSTEM = 'relative-link'

if SYSTEM == 'snellius':
    BASE_DATA_PATH = '/home/bvanberlo/data'
    BASE_DATA_PATH_LOCAL = os.environ['TMPDIR']
elif SYSTEM == 'winhpc':
    BASE_DATA_PATH = '/home/mcs001/20184025/data'
    BASE_DATA_PATH_LOCAL = '/local/20184025'
elif SYSTEM == 'grill':
    BASE_DATA_PATH = '/data-3/users/coerlema/data'
    BASE_DATA_PATH_LOCAL = BASE_DATA_PATH
else:
    # local usage
    BASE_DATA_PATH = 'Datasets'
    BASE_DATA_PATH_LOCAL = BASE_DATA_PATH  # Local tmp SSD is assumed not to exist

data_paths = {
    'signfi': {
        'gaf': {'environment': '/signfi-environment-gaf.hdf5',
                'user': '/signfi-users-gaf.hdf5'},
        'dfs': {'environment': '/signfi-environment-dfs.hdf5',
                'user': '/signfi-users-dfs.hdf5'}
    },
    'widar': {
        'gaf': '/widar3.0-domain-leave-out-dataset-gaf.hdf5',
        'dfs': '/widar3.0-domain-leave-out-dataset-benchmark-2.hdf5'
    }
}
