import sys
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton

class MyThread(QThread):
    finished = pyqtSignal()

    def __init__(self, function1, function2):
        super().__init__()
        self.function1 = function1
        self.function2 = function2

    def run(self):
        self.function1()
        self.function2()
        self.finished.emit()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Parallel Thread Example')
        layout = QVBoxLayout(self)

        self.run_button = QPushButton('Run Functions in Parallel', self)
        self.run_button.clicked.connect(self.run_functions)
        layout.addWidget(self.run_button)

    def run_functions(self):
        thread = MyThread(self.function1, self.function2)
        thread.finished.connect(self.on_thread_finished)
        thread.start()

    def function1(self):
        print("Function 1 started")
        time.sleep(3)
        print("Function 1 finished")

    def function2(self):
        print("Function 2 started")
        time.sleep(3)
        print("Function 2 finished")

    def on_thread_finished(self):
        print("Both functions finished.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
