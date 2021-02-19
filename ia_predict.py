
# derived from demo.py
# Evaluate images by doing prediction.

import os
from glob import glob
import sys
import cv2
import numpy as np
from imageio import imread, imsave
from tqdm import tqdm
import time
import argparse
import ia_work_queue as wq
import ia_postprocess as pp
import ray

from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization

# what to remove before running: ../ufilm_dataset2/page_xml ../ufilm_dataset2/processed_images 

PAGE_XML_DIR = 'page_xml'  # a subdir of output_dir
model_dir = '../ufilm_dataset3/output/export'
input_files = glob('../ufilm_testset/*.png')
input_files.sort()
output_dir = '../ufilm_testset/processed_images'
ground_file = '../ufilm_testset/ground.csv'
post_model_path = "postmodel.joblib"
enable_debug = True
Nclasses = 4  # number of labels or classes, not counting background
# when False, then use is_features and ground.csv to collect post model training data.
# When True, then use the post model to guide actions to take
production_mode = False


if __name__ == '__main__':
    FLAGS = None
    # init the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path', '-m',
        type=str,
        default='../ufilm_dataset3/output/export',
        help='path to saved model'
    )
    parser.add_argument(
        '--input_files', '-i',
        type=str,
        default='../ufilm_testset',
        help='path to dir of jpg files to process'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='../ufilm_testset/processed_images',
        help='dir to put output files'
    )
    parser.add_argument(
        '--ground_file', '-g',
        type=str,
        default='../ufilm_testset/ground.csv',
        help='ground labels for pages needed for training the post model (non-production)'
    )
    parser.add_argument(
        '--post_model_path', '-p',
        type=str,
        default='postmodel.joblib',
        help='path to post model file, ends with .joblib'
    )
    parser.add_argument(
        '--hocr_path',
        type=str,
        default="",
        help='path to hocr file, needed for production'
    )
    parser.add_argument(
        '--debug',
        action="store_true",
        help='use this to enable debug'
    )
    parser.add_argument(
        '--production',
        action="store_true",
        help='use this to to enable production mode, otherwise post-model training mode'
    )

    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print(f"  Unknown args ignored: {unparsed}")
    model_dir = FLAGS.model_path  # dhSegment model ("saved model" format)
    input_files = FLAGS.input_files  # dir of jpg or png files to process
    # glob for input_files
    input_files1 = glob(input_files + "/*.jpg")
    input_files2 = glob(input_files + "/*.png")
    input_file_list = input_files1 + input_files2
    input_file_list.sort()
    output_dir = FLAGS.output_dir  # not used in production?
    ground_file = FLAGS.ground_file  # required for post model training
    post_model_path = FLAGS.post_model_path  # required
    enable_debug = FLAGS.debug
    hocr_file = FLAGS.hocr_path
    production_mode = FLAGS.production
    if not os.path.exists(model_dir):
        print(f"model dir does not exist: {model_dir}")
        sys.exit(2)

    os.makedirs(output_dir, exist_ok=True)
    # PAGE XML format output
    output_pagexml_dir = os.path.join(output_dir, PAGE_XML_DIR)
    os.makedirs(output_pagexml_dir, exist_ok=True)

    #np.set_printoptions(precision=5)

    #
    #   start Ray, a Queue actor, and post-processing actor(s)
    #
    ray.init(dashboard_host="0.0.0.0", num_cpus=5, num_gpus=1)
    work_queue = wq.WorkQueue()
    post_process1 = pp.PostProcess.options(name="PostProcess1").remote(work_queue, output_dir, ground_file, post_model_path,
                                                                    "postmodel_training.data", enable_debug,
                                                                       production_mode, Nclasses, hocr_file)
    if production_mode:
        post_process2 = pp.PostProcess.options(name="PostProcess2").remote(work_queue, output_dir, ground_file, post_model_path,
                                                                        "postmodel_training.data", enable_debug,
                                                                           production_mode, Nclasses, hocr_file)
        post_process3 = pp.PostProcess.options(name="PostProcess3").remote(work_queue, output_dir, ground_file, post_model_path,
                                                                        "postmodel_training.data", enable_debug,
                                                                           production_mode, Nclasses, hocr_file)
        post_process4 = pp.PostProcess.options(name="PostProcess4").remote(work_queue, output_dir, ground_file, post_model_path,
                                                                        "postmodel_training.data", enable_debug,
                                                                           production_mode, Nclasses, hocr_file)
    post_process1.run.remote()
    if production_mode:
        post_process2.run.remote()
        post_process3.run.remote()
        post_process4.run.remote()

    import tensorflow as tf

    # check that GPU is present
    if not tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        print("")
        print("        WARNING:  GPU not available")
        print("")

    warned_on_file_ordering = False
    page_number_counted = 0

    with tf.Session():  # Start a tensorflow session
        # Load the model
        tf_model = LoadedModel(model_dir, predict_mode='filename')
        print("")
        print(f"          PROCESSING {len(input_file_list)} files from {input_files}")
        if production_mode:
            print("")
            print(f"                   P R O D U C T I O N")
        else:
            print("")
            print("                    Post-Model Training")
        print("")
        #  For each image
        for filename in tqdm(input_file_list, desc='Processed files'):
            basename = os.path.basename(filename).split('.')[0]
            page_number_parsed = -1
            try:
                page_number_parsed = int(basename.split("_")[-1])
            except Exception:
                pass
            start_time_sec = time.time()
            #
            #       predict each pixel's label
            #
            prediction_outputs = tf_model.predict(filename)
            finish_time_sec = time.time()
            # labels_all has shape (h,w) which is like (976, 737)
            labels_all = prediction_outputs['labels'][0]
            probs_all = prediction_outputs['probs'][0]
            # probs_all have shape like (976, 737, 4) corresponding to (H, W, Nclasses)
            original_shape = prediction_outputs['original_shape']
            if (page_number_counted != page_number_parsed):
                page_number = page_number_parsed
                if not warned_on_file_ordering:
                    warned_on_file_ordering = True
                    print(f"  WARN: page number mismatch: parse={page_number_parsed} vs. count={page_number_counted}")
            else:
                page_number = page_number_counted
            g = work_queue.group(labels_all, probs_all, filename, original_shape, finish_time_sec - start_time_sec, page_number)
            # give work to the post-processing actor
            work_queue.push(g)
            page_number_counted += 1
    print("")
    print(f"          {page_number_counted} files processed, results in {output_dir}")
    print("")
    # wait for the work_queue to be empty
    while not work_queue.empty():
        time.sleep(1.0)
    time.sleep(15.0)
    post_process1.close.remote()
    if production_mode:
        post_process2.close.remote()
        post_process3.close.remote()
        post_process4.close.remote()
