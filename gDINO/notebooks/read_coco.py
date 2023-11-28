import os
import json
import pprint

p = pprint.PrettyPrinter(indent=4)


def main():

    coco_anno_filepath = f'{data_dirpath}/coco2017/annotations/instances_train2017.json'
    # coco_anno_filepath = f'{data_dirpath}/coco2017/annotations/instances_val2017.json'

    with open(coco_anno_filepath, 'r') as f:
        data = json.load(f)
        categories = data["categories"]

    id2class = {}

    for category in categories:
        item = {int(category['id'])-1: category['name']}
        id2class.update(item)

    p.pprint(id2class)



if __name__ == '__main__':
    main()