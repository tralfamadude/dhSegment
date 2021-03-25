import json

import ray
from ray.util.queue import Queue
import ia_features
import ia_extract
import ia_post_model as pm
import ia_util as util
import time
import random
import sys
import os
import cv2
import numpy as np
from imageio import imread, imsave
from dh_segment.io import PAGE
from dh_segment.post_processing import boxes_detection, binarization
import hocr

label_bins = []  # used for histogram

# label colors
label_colors = [[0, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [1, 255, 254],
                [255, 166, 254],
                [255, 219, 102],
                [0, 100, 1]]


@ray.remote
class PostProcess:  # actor, CONSUMER of queue
    def __init__(self, work_queue: Queue, output_dir: str, ground_file: str, post_model_path: str,
                 post_model_training_data_path: str,
                 enable_debug: bool, production_mode: bool, nclasses: int, hocr_file: str = ""):
        self.work_queue = work_queue
        self.output_dir = output_dir
        self.ground_file = ground_file
        self.post_model_path = post_model_path
        self.post_model_training_data_path = post_model_training_data_path
        self.enable_debug = enable_debug
        self.production_mode = production_mode
        self.nclasses = nclasses
        self.start_time_sec = time.time()
        self.count = 0
        self.post_model = None
        self.feat = None
        self.counter = 0
        self.cleaner = util.TextUtil()
        if production_mode:
            # production means using the post model, so we load it here
            # (post_model_training_data_path is not used since it is already inherent in the post model joblib)
            self.feat = ia_features.Features(None, "dummy.data")  # no post model training
            self.post_model = pm.PostModel(post_model_path)
            if len(hocr_file) == 0:
                print("PostProcess must have an hocr file in production mode")
                raise NameError("PostProcess must have an hocr file in production mode")
            self.extractor = ia_extract.ExtractOCR(hocr_file)
        else:
            # training mode for post model
            self.feat = ia_features.Features(ground_file, post_model_training_data_path)
            self.extractor = None
        for i in range(0, nclasses + 1):
            label_bins.append(i)  # [0] is background, not a real label
        if not production_mode:
            #   make a png to show which colors map to which labels (top bar is background)
            label_color_demo = np.zeros([20 * (nclasses + 1), 200, 3], np.uint8)
            for labeli in range(0, nclasses + 1):
                c = self.label_val_to_color(labeli)
                for h in range(labeli * 20, 20 + (labeli * 20)):
                    for w in range(0, 200):
                        label_color_demo[h, w, 0] = c[0]
                        label_color_demo[h, w, 1] = c[1]
                        label_color_demo[h, w, 2] = c[2]
            imsave(os.path.join(output_dir, "label_demo.png"), label_color_demo)
        # Save txt file
        # with open(os.path.join(output_dir, 'pages.txt'), 'w') as f:
        #    f.write(txt_coordinates)
        self.results_log_path = os.path.join(output_dir, f"info{random.randint(1, 1000)}.log")
        self.results_log = open(self.results_log_path, "a")

    def close(self):
        self.feat.close()
        self.results_log.flush()
        self.results_log.close()

    def _put_results_log(self, s):
        # NOTE: only one pp worker can do this
        self.results_log.write(f"{s}\n")
        self.results_log.flush()

    def get_count(self):
        """
        :return: count of files processed.
        """
        return self.counter

    def remove_author_superscripts(authors):  # unused
        """
        :return:
        """
        pass

    def get_uptime(self):
        """
        :return: uptime in seconds, as a float.
        """
        return time.time() - self.start_time_sec

    def label_val_to_color(self, labelv):
        """
        Map a label number to a color.
        :param labelv: label value, 0-Nclasses inclusive where 0 is background.
        :return: [r,g,b]
        """
        return label_colors[labelv]

    def page_make_binary_mask(self, probs: np.ndarray, threshold: float = -1) -> np.ndarray:
        """
        Computes the binary mask of the detected Page from the probabilities outputed by network
        :param probs: array with values in range [0, 1]
        :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
        :return: binary mask
        """

        mask = binarization.thresholding(probs, threshold)
        mask = binarization.cleaning_binary(mask, kernel_size=5)
        return mask

    def format_quad_to_string(self, quad):
        """
        Formats the corner points into a string.
        :param quad: coordinates of the quadrilateral
        :return:
        """
        s = ''
        for corner in quad:
            s += '{},{},'.format(corner[0], corner[1])
        return s[:-1]

    async def run(self):
        while True:
            #
            #   get item off work queue
            #
            start_wait = time.time()
            g = self.work_queue.pop()
            finish_wait = time.time()
            self.counter += 1
            labels_all, probs_all, filename, original_shape, inference_time_sec, page_number = self.work_queue.ungroup(g)
            basename = os.path.basename(filename).split('.')[0]
            self.feat.start(basename)
            if self.enable_debug:
                # write out an image of the per pixel labels
                label_viz = np.zeros((labels_all.shape[0], labels_all.shape[1], 3), np.uint8)
                for h in range(0, labels_all.shape[0]):
                    for w in range(0, labels_all.shape[1]):
                        c = self.label_val_to_color(labels_all[h, w])
                        label_viz[h, w, 0] = c[0]
                        label_viz[h, w, 1] = c[1]
                        label_viz[h, w, 2] = c[2]
                imsave(os.path.join(self.output_dir, f"{basename}_label_viz.png"), label_viz)
            # what pixel labels do we have?
            hist_label_counts = np.bincount(labels_all.flatten()).tolist()
            while len(hist_label_counts) < max(label_bins) + 1:
                hist_label_counts.append(0)
            # now hist_label_counts contains counts of pixel labels

            self._put_results_log(f"processing: file={filename} histogram={hist_label_counts}  "
                                  f"infer_timing={inference_time_sec} original_shape={original_shape}")

            original_img = imread(filename, pilmode='RGB')
            if self.enable_debug:
                original_img_box_viz = np.array(original_img)
                original_img_box_viz_modified = False

            #
            #    handle rectangles here!
            #
            for label_slice in label_bins:
                if label_slice == 0:
                    continue  # skip background
                color_tuple = self.label_val_to_color(label_slice)
                #  area of all the pixel labels for a particular class, might be multiple regions
                area = hist_label_counts[label_slice]
                if area < 500:  # minimum size
                    # reject small label areas
                    continue

                probs = probs_all[:, :, label_slice]

                #        make an image showing probability map for this label before postprocessing
                #            (it can include multiple blobs)
                if self.enable_debug:
                    prob_img = np.zeros((probs.shape[0], probs.shape[1], 3), np.uint8)
                    for h in range(0, probs.shape[0]):
                        for w in range(0, probs.shape[1]):
                            c = probs[h, w] * 255
                            prob_img[h, w, 0] = c
                            prob_img[h, w, 1] = c
                            prob_img[h, w, 2] = c
                    imsave(os.path.join(self.output_dir, f"{basename}_{label_slice}_label_prob.png"), prob_img)

                # Binarize the predictions
                page_bin = self.page_make_binary_mask(probs)

                # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
                bin_upscaled = cv2.resize(page_bin.astype(np.uint8, copy=False),
                                          tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)
                # upscale probs the same way so we can calculate confidence later
                probs_upscaled = cv2.resize(probs.astype(np.float32, casting='same_kind'),
                                            tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)

                # Find quadrilateral(s) enclosing the label area(s).
                #  allow more than reasonable number of boxes so we can use spurious boxes as a reject signal
                pred_region_coords_list = boxes_detection.find_boxes(bin_upscaled.astype(np.uint8, copy=False),
                                                                     mode='rectangle', min_area=0.001, n_max_boxes=4)

                # coord is [[a,b], [c,b], [c,d], [a,d]]  (a path for drawing a polygon, clockwise)
                #  origin is upper left [x,y]:
                #  [a,b]         [c,b]
                #       rectangle
                #  [a,d]         [c,d]
                # which means a<c and b<d

                if pred_region_coords_list is not None:
                    # Draw region box on original image and export it. Add also box coordinates to the txt file
                    region_count = len(pred_region_coords_list)
                    count = 0
                    for pred_region_coords in pred_region_coords_list:
                        #  cut out rectangle for region based on original image size
                        a = pred_region_coords[0, 0]
                        b = pred_region_coords[0, 1]
                        c = pred_region_coords[1, 0]
                        d = pred_region_coords[2, 1]
                        probs_rectangle = probs_upscaled[b:d + 1, a:c + 1]  # values are in range [0,1]
                        overall_confidence = (sum(sum(probs_rectangle))) / ((c - a) * (d - b))
                        aspect_ratio = (c - a) / (d - b)  # w/h
                        page_width_fraction = (c - a) / original_shape[0]
                        page_height_fraction = (d - b) / original_shape[1]
                        normalized_x = a / original_shape[0]
                        normalized_y = b / original_shape[1]
                        region_size = page_width_fraction * page_height_fraction
                        cmts = f"Prediction {a},{b},{c},{d} confidence={overall_confidence} aspect={aspect_ratio} widthfrac={page_width_fraction} heightfrac={page_height_fraction} normalized_x={normalized_x} normalized_y={normalized_y} dimensions={c - a}x{d - b} spec={basename}_{label_slice}-{count}"
                        self._put_results_log(cmts)
                        img_rectangle = original_img[b:d + 1, a:c + 1]
                        tag_rect_x0 = a
                        tag_rect_y0 = b
                        tag_rect_x1 = c
                        tag_rect_y1 = d
                        if self.enable_debug:
                            # draw box to visualize rectangle
                            cv2.polylines(original_img_box_viz, [pred_region_coords[:, None, :]], True,
                                          (color_tuple[0], color_tuple[1], color_tuple[2]), thickness=5)
                            original_img_box_viz_modified = True
                            imsave(os.path.join(self.output_dir,
                                                f"{basename}_{label_slice}-{count}_{overall_confidence}_rect.jpg"),
                                   img_rectangle)
                        # Write corners points into a .txt file
                        # txt_coordinates += '{},{}\n'.format(filename, self.format_quad_to_string(pred_region_coords))

                        # store info on area for use after all areas in image are gathered
                        self.feat.put(label_slice, count, region_size, overall_confidence, aspect_ratio,
                                      page_width_fraction, page_height_fraction,
                                      normalized_x, normalized_y,
                                      tag_rect_x0, tag_rect_y0, tag_rect_x1, tag_rect_y1,
                                      img_rectangle, cmts)

                        # Create page region and XML file
                        page_border = PAGE.Border(coords=PAGE.Point.cv2_to_point_list(pred_region_coords[:, None, :]))

                        count += 1
                else:
                    # No box found for label
                    # page_border = PAGE.Border()
                    continue
            if self.enable_debug:
                # boxes for all labels, using mask colors
                if original_img_box_viz_modified:
                    imsave(os.path.join(self.output_dir, f"{basename}__boxes.jpg"), original_img_box_viz)

            self.feat.finish()  # finish image, in non-production this saves feature vector for post model
            page_prediction_msg = ""
            prediction_summary_txt = ""
            if self.production_mode:
                #
                #    apply post-model to determine page type
                #
                v = np.zeros((1, self.feat.vec_length()))
                v[0] = self.feat.get_post_model_vec()
                y = self.post_model.predict(v)
                page_type = int(y[0])
                page_prediction_msg = f"PagePrediction: {basename} "

                #
                #    take actions
                #
                if page_type == 0:                           # other page, skip
                    page_prediction_msg += f"type=0"
                    pass
                elif page_type == 1:                         # start page of article, save info
                    page_prediction_msg += f"type=1"
                    title_info = self.feat.get_label_instance(1, 0)
                    title_rect_x0 = 2 * title_info["tag_rect_x0"]
                    title_rect_y0 = 2 * title_info["tag_rect_y0"]
                    title_rect_x1 = 2 * title_info["tag_rect_x1"]
                    title_rect_y1 = 2 * title_info["tag_rect_y1"]
                    title_normalized_y = title_info["normalized_y"]

                    author_info = self.feat.get_label_instance(2, 0)
                    author_rect_x0 = 2 * author_info["tag_rect_x0"]
                    author_rect_y0 = 2 * author_info["tag_rect_y0"]
                    author_rect_x1 = 2 * author_info["tag_rect_x1"]
                    author_rect_y1 = 2 * author_info["tag_rect_y1"]
                    author_normalized_y = author_info["normalized_y"]

                    acceptable = True
                    # qualifications
                    if title_info["confidence"] < .5 or author_info["confidence"]:
                        # too low, could be 0 (missing)
                        msg = f"  REJECT confidence too low  "
                        self._put_results_log(msg)
                        prediction_summary_txt += msg + "\n"
                        acceptable = False
                    if title_rect_y0 > author_rect_y0:
                        # unusual, author appears above title
                        msg = f"  REJECT author appears above title  "
                        self._put_results_log(msg)
                        prediction_summary_txt += msg + "\n"
                        acceptable = False
                    if title_normalized_y > 0.5 or author_normalized_y > 0.5:
                        msg = f"  REJECT: title or author appears in lower half of page "
                        self._put_results_log(msg)
                        prediction_summary_txt += msg + "\n"
                        acceptable = False
                    title = self.extractor.find_bbox_text(page_number, title_rect_x0, title_rect_y0, title_rect_x1, title_rect_y1)
                    title = self.cleaner.one_line(title)
                    authors = self.extractor.find_bbox_text(page_number, author_rect_x0, author_rect_y0, author_rect_x1, author_rect_y1)
                    authors = self.cleaner.cleanAuthors(authors)
                    smsg = f"{basename}: page={page_number} TITLE={title} AUTHORS={authors}"
                    self._put_results_log(smsg)
                    prediction_summary_txt += smsg
                    prediction_summary_txt += f"\nTITLE({title_info['comments']})\n"
                    prediction_summary_txt += f"AUTHOR({title_info['comments']})\n"
                    if acceptable:
                        json_per_image = os.path.join(self.output_dir, f"{basename}.json")
                        json_dict = {}
                        json_dict["page_number"] = page_number
                        json_dict["basename"] = basename
                        json_dict["type"] = "start_article"
                        json_dict["title"] = title
                        json_dict["authors"] = authors
                        json_txt = json.dumps(json_dict)
                        with open(json_per_image, "a") as f:
                            f.write(f"{json_txt}\n")
                elif page_type == 2:                              # references page, save info
                    page_prediction_msg += f"type=2"
                    pass
                else:                                             #  toc page, save info
                    # pagte_type == 3
                    page_prediction_msg += f"type=3"
                    pass
            else:  # mode for gathering of training data for post model
                pass
            finish_post = time.time()
            self._put_results_log(
                f"TIMING: wait={finish_wait - start_wait} post={finish_post - finish_wait} {page_prediction_msg}")

            #   if debug, emit a txt summary
            if self.enable_debug:  ##############################
                if len(prediction_summary_txt) > 0:
                    debug_per_image = os.path.join(self.output_dir, f"{basename}.txt")
                    with open(debug_per_image, "a") as f:
                        f.write(f"{prediction_summary_txt}\n")
