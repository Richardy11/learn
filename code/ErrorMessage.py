import matplotlib
import matplotlib.pyplot as plt

from matplotlib.widgets import Button

def close(event):
    plt.close()


failfig = plt.figure()
mngr = plt.get_current_fig_manager()
axis1 = failfig.add_subplot(111)
mngr.window.setGeometry(500,100,600,300)
plt.xlim(-1,1)
plt.ylim(-1,1)
axis1.axis('off')
axis1.text(0,0.6,'You cannot choose the same \nmovement for multiple DoFs.\nPlease try again.',ha='center',va='center',fontsize = 20)
buttonax = plt.axes([0.3,0.2,0.4,0.15])
button = Button(buttonax,'CLOSE')
button.on_clicked(close)
plt.show()