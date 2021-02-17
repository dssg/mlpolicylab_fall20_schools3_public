import io
from schools3.data.base.processor import Processor
from schools3.config.data import db_config


class Explorer(Processor):
    def explore(self):
        '''
            abstract function that can be overridden to contain
            the bulk of the analysis in derived classes
        '''
        return NotImplementedError()
