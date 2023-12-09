import os
import yaml


class ConfigParser(object):
    config_file = os.path.dirname(os.path.realpath(__file__)) + '/config.yaml'
    configs = yaml.load(open(config_file, 'r'), yaml.FullLoader)

    @classmethod
    def get(cls, server='config', key=None):
        if not cls.configs:
            cls.configs = yaml.load(open(cls.config_file, 'r'))
        section = cls.configs.get(server, None)
        if section is None:
            raise NotImplementedError
        value = section.get(key, None)
        if value is None:
            raise NotImplementedError
        return value
