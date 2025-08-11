from custom_items.data_fetching import get_input_output_shape
from custom_items.tuners import get_hyperband_tuner
from tuning_supervised import get_build_model_function


def print_best_results(model_name, dataset, datatype, backbone, signfi_experiment):
    print('best results:', model_name, dataset, datatype)
    input_shape, output_shape, domain_shape = get_input_output_shape(dataset, datatype, signfi_experiment,
                                                                     return_domain=True)

    build_model = get_build_model_function(model_name, datatype, backbone, input_shape, output_shape,
                                           domain_shape)

    tuner = get_hyperband_tuner(build_model, model_name, datatype)
    tuner.reload()
    best_trail = tuner.oracle.get_best_trials()[0]
    best_trail.summary()
    print(best_trail.get_state())


print_best_results('wigrunt', 'widar', 'gaf', 'efficientnet', None)
print_best_results('wigrunt', 'widar', 'dfs', 'efficientnet', None)
