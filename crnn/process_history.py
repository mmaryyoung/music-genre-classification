import matplotlib.pyplot as plt

# Takes in a Keras history callback from a model fitting and plots the past run's loss and accuracy
# into a figure and saves it locally.
def plot_history(history, title):
  training_accuracy = history.history['categorical_accuracy']
  training_loss = history.history['loss']
  validation_accuracy = history.history['val_categorical_accuracy']
  validation_loss = history.history['val_loss']
  plt.plot(training_accuracy, label='training_acc', color='orange', linestyle='-')
  plt.plot(training_loss, label='training_loss', color='red', linestyle=':')
  plt.plot(validation_accuracy, label='val_acc', color='olive', linestyle='-')
  plt.plot(validation_loss, label='val_loss', color='green', linestyle=':')
  plt.legend()
  plt.title(title)
  filename =  title + '.png'
  plt.savefig(filename.replace(' ', '_'))
  plt.clf()
