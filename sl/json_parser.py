from dacite import from_dict
from rts_wrapper.datatypes import Records, Record
import json
import os

from .utils import store

base_dir = '~/'
path = os.path.expanduser(os.path.join(base_dir, "records"))
saving_dir = os.path.expanduser(os.path.join(base_dir, "rcds_rvr.pck"))


def process():
    records_list = []
    cnt = 0
    for filename in os.listdir(path):
        if cnt > 480:
            break
        cnt += 1
        print(cnt)
        if cnt % 100 == 0:
            print('Game No.{} processed'.format(cnt))
        json_arr = json.load(open(os.path.join(path,filename))) 
        records = json_arr['records']
        for record in records:
            r = from_dict(data_class=Record, data=record)
            records_list.append(r)
        
    store(records_list, saving_dir)

if __name__ == "__main__":
    process()
    