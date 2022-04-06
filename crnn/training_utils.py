import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adamax, Adam, Nadam, RMSprop, SGD, Ftrl
from model_utils import createCRNNModel
import itertools
from operator import itemgetter
from tabulate import tabulate

# All potential network configurations.
OPT_TYPES = [
    'adam',
    'nadam',
    'rmsprop',
    'sgd',
    'adadelta',
    'adagrad',
    'adamax',
    'ftrl',
    ]

LEARNING_RATES = [
    1,
    0.1,
    0.01,
    0.001,
    0.0001]

CONV_NUMS = [
    1,
    2,
    3]

CONV_FILTERS = [
    32,
    64,
    128]

CONV_KERENEL_SIZES = [
    3,
    5,
    10]

CONV_STRIDES = [
    # 1,
    3,
    5]

# Translates string names for optimizers into TF optimizer objects.
def _get_optimizer(opt_type, learning_rate):
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

# Returns the cartesian product of the incoming arguments (lists)
# as a dictionary, where the keys are the arguments' names.
def _product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

# Returns a list of dictionaries that each contains the config name-to-value
# mapping. This list is the cross product of all config dimensions.
# Dictionary keys include:
#   opt_type, learning_rate, conv_num, conv_filter, conv_kernel_size, conv_stride
def get_all_configs():
    all_configs = _product_dict(
        opt_type=OPT_TYPES,
        learning_rate=LEARNING_RATES,
        conv_num=CONV_NUMS,
        conv_filter=CONV_FILTERS,
        conv_kernel_size=CONV_KERENEL_SIZES,
        conv_stride=CONV_STRIDES)
    return filter(lambda c: not is_ignore_config(c), all_configs)

# Stores a list of "to be ignored" config combinations
# and checks the current config against that list. 
def is_ignore_config(config):
    ignore_predicates = [
        lambda c: c['opt_type'] == 'adam' and c['learning_rate'] > 0.001,
        lambda c: c['opt_type'] == 'nadam' and c['learning_rate'] > 0.001,
        lambda c: c['opt_type'] == 'rmsprop' and c['learning_rate'] > 0.0001,
        lambda c: c['opt_type'] == 'sgd' and (c['learning_rate'] > 0.01 or c['learning_rate'] < 0.01), 
        lambda c: c['opt_type'] == 'adadelta' and c['learning_rate'] < 0.1,
        lambda c: c['opt_type'] == 'adagrad' and (c['learning_rate'] != 0.01),
        lambda c: c['opt_type'] == 'adamax' and c['learning_rate'] > 0.001,
        lambda c: c['opt_type'] == 'sgd' and c['conv_kernel_size'] == 10 and c['conv_stride'] == 3,
        lambda c: c['opt_type'] == 'ftrl' and (c['learning_rate'] != 0.1 or c['conv_num'] > 1),
        lambda c: c['conv_kernel_size'] < c['conv_stride']
    ]
    for p in ignore_predicates:
        if p(config):
            return True
    return False

# Converts dictionary to string with the key1=val1-key2=val2 format.
def dict_to_string(dictionary):
    return '-'.join([k + '=' + str(v) for k, v in dictionary.items()])

# Trains the global data with variable optimizer specs. Note that the input x_train
# should be of shape (num_samples, num_mels, num_chunks_per_sample, 1)
def train_with_config(x_tr, y_tr, x_te, y_te, opt_type, learning_rate, conv_num, conv_filter, conv_kernel_size, conv_stride):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=7)
    opt = _get_optimizer(opt_type, learning_rate)
    model = createCRNNModel(
        x_tr.shape[1:],
        num_classes=y_tr.shape[-1],
        conv_num=conv_num,
        conv_filter=conv_filter,
        conv_kernel_size=conv_kernel_size,
        conv_stride=conv_stride)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    history = model.fit(
        x=x_tr, y=y_tr,
        validation_data=(x_te, y_te),
        batch_size=10,
        epochs=100,
        callbacks=[early_stopping_callback],
        verbose=2,
        shuffle=True)
    return history, model

# Same as train_with_config, but with an existing model as a starting point.
def train_with_model(x_tr, y_tr, x_te, y_te, model, opt_type, learning_rate):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=7)
    opt = _get_optimizer(opt_type, learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    history = model.fit(
        x=x_tr, y=y_tr,
        validation_data=(x_te, y_te),
        batch_size=10,
        epochs=100,
        callbacks=[early_stopping_callback],
        verbose=2,
        shuffle=True)
    # Prints and saves the best result from this config.
    max_acc_idx, max_acc = max(enumerate(history.history['val_categorical_accuracy']), key=itemgetter(1))
    return model

def train_with_all_configs(all_configs, x_tr, y_tr, x_te, y_te):
    results_table = []
    best_model = None
    best_val_acc = 0
    for config in all_configs:
        config_summary_str = dict_to_string(config)
        print('Training with model: %s.' % config_summary_str)
        try:
            history, current_model = train_with_config(x_tr, y_tr, x_te, y_te, **config)
            # Prints and saves the best result from this config.
            max_acc_idx, max_acc = max(enumerate(history.history['val_categorical_accuracy']), key=itemgetter(1))
            print(
                'Done with the %s model. Best validation accuracy is %f, achieved at epoch #%d.' % (
                config_summary_str, max_acc, max_acc_idx + 1))
            results_table.append({
                **config,
                'best_val_accuracy': max_acc,
                'best_val_accuracy_epoch': max_acc_idx + 1
            })
            if max_acc > best_val_acc:
                best_model = current_model
                best_val_acc = max_acc
        except ValueError as e:
            print('The config %s does not work. Error: %s.' % (config_summary_str, e))
    print("Done trying all configs. Here is the results table:")
    print(tabulate(results_table, headers='keys', tablefmt='github'))
    return best_model