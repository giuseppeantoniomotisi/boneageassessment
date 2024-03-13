from tools import Preprocessing

def process():
    Preprocessing().preprocessing_directory()

def single_process():
    Preprocessing().preprocessing_image('1377.png',save=False, show=False)

if __name__ == '__main__':
    single_process()
