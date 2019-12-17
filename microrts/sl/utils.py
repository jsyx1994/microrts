import dill
from microrts.rts_wrapper.envs.datatypes import Records

def store(records, path):
    """
    store records to disk
    :param path:
    :param records: list of records
    :return:
    """

    with open(path, 'wb') as f:
        records = Records(records)
        dill.dump(records, f, recurse=True, protocol=dill.HIGHEST_PROTOCOL)
    # f = open("/home/toby/rcds.pck", 'rb')
    # rcd = dill.load(f)
    # print(rcd.records.__len__())

def load(path) -> Records:
    """
    loads data from the specific path
    """
    with open(path, 'rb') as f:
        rcd = dill.load(f)
    return rcd