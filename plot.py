import os
os.environ['MPLBACKEND'] = 'TkAgg'
import matplotlib.pyplot as plt
test_loss=[1.6,1.3,1.4,1.0,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.32,0.31,0.35,0.32,0.32,0.32,0.31,0.35,0.31,0.31]
epochs = len(test_loss)
epochs_vec = range(epochs)
plt.plot(epochs_vec,test_loss,label="test_loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()
plt.show()