import matplotlib.pyplot as plt

def plot_history(history, title):
  training_accuracy = history.history['categorical_accuracy']
  training_loss = history.history['loss']
  validation_accuracy = history.history['val_categorical_accuracy']
  validation_loss = history.history['val_loss']
  plt.plot(training_accuracy, label='training_acc', linestyle='-')
  plt.plot(training_loss, label='training_loss', linestyle=':')
  plt.plot(validation_accuracy, label='val_acc', linestyle='-')
  plt.plot(validation_loss, label='val_loss', linestyle=':')
  plt.legend()
  plt.title(title)
  plt.savefig(title + '.png')
  return plt