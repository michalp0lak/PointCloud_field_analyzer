import os

class global_Parameters(object):

    def __init__(self):

        """
        Parameter LASFILE_PATH navigates software in folder, where target las file is
        stored and where output files of this module will be generated.
        """
        self.LASFILE_PATH = '.....'

        """
        Parameter FILENAME determines name of processed las file.
        """
        self.FILENAME = '.....'

        """
        Parameter PLOT_NUM is number of experimental blocks in field.
        """
        self.BLOCK_NUM = 5

        """
        Parameter SUBPLOT_NUM is number of experimental units in single experimental block
        """
        self.PLOT_NUM = 53
