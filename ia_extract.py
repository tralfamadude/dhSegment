import hocr

import time


# "/Users/peb/Downloads/sim_bjog_2013-05_120_6_hocr.html"

class ExtractOCR:
    def __init__(self, hocr_file):
        self.hocr_file = hocr_file
        self.page_iter = hocr.parse.hocr_page_iterator(self.hocr_file)
        self.page_offset = 0
        self.iter_value = self.page_iter.__next__()

    def next_page(self):
        self.iter_value = self.page_iter.__next__()
        self.page_offset += 1

    def seek_page(self, offset):
        if self.page_offset > offset:
            raise IndexError
        while self.page_offset < offset:
            self.next_page()

    def find_bbox_text(self, offset, x0, y0, x1, y1):
        """

        :param offset: page offset from which to extract text
        :param x0: left edge
        :param y0: top edge
        :param x1: right edge
        :param y1: top edge
        :return: text found in bounding box.
        """
        ret = ""
        self.seek_page(offset)
        #w, h = hocr.parse.hocr_page_get_dimensions(self.iter_value)
        #print(f" page {offset} is ({w}, {h})")
        word_data = hocr.parse.hocr_page_to_word_data(self.iter_value)
        for paragraph in word_data:
            for line in paragraph['lines']:
                for word in line['words']:
                    if x0 <= word['bbox'][0] and x1 >= word['bbox'][2] and y0 <= word['bbox'][1] and y1 >= word['bbox'][3]:
                        ret += word['text']
                        ret += " "
                ret += "\n"
        return ret.strip()

"""
Demonstration:
"""
if __name__ == '__main__':
    eocr = ExtractOCR("/Users/peb/Downloads/sim_bjog_2013-05_120_6_hocr.html")
    # dump entire page 15 (starts from 0)
    #text = eocr.find_bbox_text(15, 0., 0.0, 3322.0, 4300.0)  #  entire page

    # page 8 has TOC, 748,173,1368,1120    and   109,173,721,1127  (need to double these values)
    #text = eocr.find_bbox_text(8, 1496,  346, 2736, 2240)
    #print(f"{text}")
    #text = eocr.find_bbox_text(8, 218,  346, 1442, 2254)
    #print(f"{text}")

    # page 38 has first page of article
    #   title: [402,  500, 2596,  882]
    #   authors: [410,  900, 2038,  994]
    text = eocr.find_bbox_text(38, 402,  500, 2596,  882)
    print(f" TITLE:  {text}")
    text = eocr.find_bbox_text(38, 410,  900, 2038,  994)
    print(f" AUTHORS:  {text}")

