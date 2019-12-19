from dacite import from_dict
from microrts.rts_wrapper.envs.datatypes import Records, Record
import json
import os
import microrts.settings as settings

from .utils import store
import microrts.settings as settings

base_dir = '~/'
path = os.path.expanduser(os.path.join(base_dir, "records"))
saving_dir = settings.data_dir

# TODO: multi-threading
def process():
    records_list = []
    cnt = 0
    for filename in os.listdir(path):
        cnt += 1
        print(cnt)
        if cnt > 500:
            break
        if cnt % 100 == 0:
            print('Game No.{} processed'.format(cnt))
        with open(os.path.join(path,filename)) as f:
            json_arr = json.load(f)
            records = json_arr['records']
            for record in records:
                r = from_dict(data_class=Record, data=record)
                records_list.append(r)
        
    store(records_list, os.path.join(settings.data_dir, "rvr6x6.pck"))

if __name__ == "__main__":
    process()
    