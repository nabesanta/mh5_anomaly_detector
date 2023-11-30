import numpy as np
import os

class DataPersist:
    def __init__(self):
        pass

    def __store_data_loc(self, name):
        path_parts = name.split('.')[0].split('/')
        file_name = path_parts[-1]
        path = ['mod_data' if x == 'data' else x for x in path_parts[:-1]]
        path = '/'.join(path)
        loc = f"{path}/{file_name}.npz"

        if not os.path.exists(os.path.abspath(path)):
            os.makedirs(os.path.abspath(path))

        return loc

    def dump_data(self, name, x, y):
        loc = self.__store_data_loc(name)
        np.savez(open(loc, 'w+b'), x=x, y=y)
        print('Saved to', loc)

    def retrieve_dumped_data(self, name):
        loc = self.__store_data_loc(name)
        data = np.load(loc)
        x = data['x']
        y = data['y']
        data.close()
        return x, y
