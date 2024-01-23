import os
import cv2
import glob
import shutil
import random
import logging
import argparse
import numpy as np
from tqdm import trange, tqdm
from datetime import timedelta

import dv_processing as dv
import dv_toolkit as kit

import xml.etree.ElementTree as ET 


def create_folder(path, needs_clean=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif needs_clean:
        shutil.rmtree(path)
        os.makedirs(path)
    return path


def process_data(args, file):
    # initialize folder
    clip_path  = create_folder(os.path.join(file, 'clips'),  args.replace)
    image_path = create_folder(os.path.join(file, 'images'), args.replace)

    # load offline data
    reader = kit.io.MonoCameraReader(f"{file}/record.aedat4")
    data, resolution = reader.loadData(), reader.getResolution("events")

    # register a slicer
    if args.clip:
        slicer = kit.MonoCameraSlicer()

    # register accumulator
    if args.image:
        accumulator = dv.Accumulator(resolution)
        accumulator.setMinPotential(-np.inf)
        accumulator.setMaxPotential(+np.inf)
        accumulator.setEventContribution(1.0)
        accumulator.setIgnorePolarity(False)
        accumulator.setDecayFunction(dv.Accumulator.Decay.NONE)

    # initialize
    clip_index = 0
    if not data.frames().isEmpty():
        latest_frame = data.frames().at(0).image
    else:
        latest_frame = np.zeros(resolution[::-1]).astype(np.uint8)

    def subprocess(data):
        # define nonlocal variable
        nonlocal clip_index, latest_frame

        # set output file name
        output_name = f"{clip_index:05d}"
        
        # update for next subprocess
        clip_index = clip_index + 1
        latest_frame = data.frames().at(0).image if not data.frames().isEmpty() else latest_frame
        
        if args.clip:
            writer = kit.io.MonoCameraWriter(f"{clip_path}/{output_name}.aedat4", resolution)
            writer.writeData(data)

        if args.image:
            # generate count image
            accumulator.clear()
            accumulator.accept(data.events().toEventStore())
            count = accumulator.getPotentialSurface()

            # overlap to image
            image = cv2.cvtColor(latest_frame.copy(), cv2.COLOR_GRAY2RGB)
            image[count != 0] = 0
            image[count > 0, 2] = 255
            image[count < 0, 1] = 255

            cv2.imwrite(f"{image_path}/{output_name}.png", image)

    # do every 33ms (cannot modify!)
    slicer = kit.MonoCameraSlicer()
    slicer.doEveryTimeInterval("events", timedelta(milliseconds=33), subprocess)
    slicer.accept(data)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="paramters")
    parser.add_argument('--clip', action="store_false")
    parser.add_argument('--image', action="store_false")
    parser.add_argument('--replace', action="store_false")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    # initialize file lists and output directories
    cwd = os.path.dirname(__file__)
    input_path = os.path.join(cwd, f'data/')

    # register a logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        handlers=[logging.FileHandler('output.log'), logging.StreamHandler()])
    logging.info(f"Now loading {input_path}")
    logging.info(f"{args}")

    # recursively obtain file list
    element_list = []
    for file in tqdm(sorted(glob.glob(f"{input_path}/*"))):
        process_data(args, file)

        # parse annotations
        tree = ET.parse(os.path.join(file, 'annotations.xml'))
        for image in tree.getroot().findall('.//image'):
            # reset name
            original_name = image.get('name').split('/')
            modified_name = f"{original_name[0]}/clips/{original_name[1][9:14]}.aedat4"
            image.set('name', modified_name)

            # update element
            element_list.append(image)

    # random shuffle and split
    random.seed(args.seed)
    random.shuffle(element_list)
    ratio_index = int(0.8 * len(element_list))
    train_set   = element_list[:ratio_index]
    test_set    = element_list[ratio_index:]
    annotations = ET.Element("annotations")
    
    # append elements to train
    train_elem = ET.SubElement(annotations, 'train')
    for id, image_elem in enumerate(train_set):
        image_elem.set('id', f"{id}")
        train_elem.append(image_elem)

    # append elements to test
    test_elem = ET.SubElement(annotations, 'test')
    for id, image_elem in enumerate(test_set):
        image_elem.set('id', f"{id}")
        test_elem.append(image_elem)

    # write to xml file
    shuffled_tree = ET.ElementTree(annotations)
    shuffled_tree.write('shuffled_file_list.xml')
