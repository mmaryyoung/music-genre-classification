from tensorflow.keras.optimizers import Adadelta, Adagrad, Adamax, Adam, Nadam, RMSprop, SGD, Ftrl
import itertools

# Translates string names for optimizers into TF optimizer objects.
def get_optimizer(opt_type, learning_rate):
    if opt_type == 'adam':
        return Adam(learning_rate=learning_rate)
    elif opt_type == 'nadam':
        return Nadam(learning_rate=learning_rate)
    elif opt_type == 'sgd':
        return SGD(learning_rate=learning_rate)
    elif opt_type == 'adadelta':
        return Adadelta(learning_rate=learning_rate)
    elif opt_type == 'adagrad':
        return Adagrad(learning_rate=learning_rate)
    elif opt_type == 'adamax':
        return Adamax(learning_rate=learning_rate)
    elif opt_type == 'ftrl':
        return Ftrl(learning_rate=learning_rate)
    elif opt_type == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    else:
        raise NotImplementedError('Optimizer type %s is not implemented.' % opt_type)

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def ignore_config(config):
    ignore_predicates = [
        lambda c: c['opt_type'] == 'adam' and c['learning_rate'] > 0.001,
        lambda c: c['opt_type'] == 'nadam' and c['learning_rate'] > 0.001,
        lambda c: c['opt_type'] == 'rmsprop' and c['learning_rate'] > 0.0001,
        lambda c: c['opt_type'] == 'sgd' and (c['learning_rate'] > 0.01 or c['learning_rate'] < 0.01), 
        lambda c: c['opt_type'] == 'adadelta' and c['learning_rate'] < 0.1,
        lambda c: c['opt_type'] == 'adagrad' and (c['learning_rate'] != 0.01),
        lambda c: c['opt_type'] == 'adamax' and c['learning_rate'] > 0.001,
        lambda c: c['opt_type'] == 'sgd' and c['conv_kernel_sizes'] == 10 and c['conv_strides'] == 3,
        lambda c: c['opt_type'] == 'ftrl' and (c['learning_rate'] != 0.1 or c['conv_nums'] > 1),
        lambda c: c['conv_kernel_sizes'] < c['conv_strides']
    ]
    for p in ignore_predicates:
        if p(config):
            return True
    return False