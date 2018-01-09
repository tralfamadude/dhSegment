from xml.etree import ElementTree as ET
from typing import List, Optional, Union, Tuple
import numpy as np
import datetime

_ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
_attribs = {'xmlns:xsi': "http://www.w3.org/2001/XMLSchema-instance",
            'xsi:schemaLocation': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 "
                                  "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"}


def _try_to_int(d: Optional[Union[str, int]])-> Optional[int]:
    if isinstance(d, str):
        return int(d)
    else:
        return d


def _get_text_equiv(e: ET.Element) -> str:
    tmp = e.find('p:TextEquiv', _ns)
    if tmp is None:
        return ''
    tmp = tmp.find('p:Unicode', _ns)
    if tmp is None:
        return ''
    return tmp.text


class Point:
    def __init__(self, y: int, x: int):
        self.y = y
        self.x = x

    @classmethod
    def list_from_xml(cls, e: ET.Element) -> List['Point']:
        if e is None:
            print('warning, trying to construct list of points from None, defaulting to []')
            return []
        t = e.attrib['points']
        result = []
        for p in t.split(' '):
            values = p.split(',')
            assert len(values) == 2
            x, y = int(values[0]), int(values[1])
            result.append(Point(y,x))
        return result

    @classmethod
    def list_to_cv2poly(cls, list_points: List['Point']):
        return np.array([(p.x, p.y) for p in list_points], dtype=np.int32).reshape([-1, 1, 2])

    @classmethod
    def cv2_to_point_list(cls, cv2_array) -> List['Point']:
        return [Point(p[0, 0], p[0, 1]) for p in cv2_array]

    @classmethod
    def list_point_to_string(cls, list_points: List['Point']):
        return ' '.join(['{},{}'.format(p.y, p.x) for p in list_points])


class BaseElement:
    tag = None

    @classmethod
    def full_tag(cls):
        return '{{{}}}{}'.format(_ns['p'], cls.tag)

    @classmethod
    def check_tag(cls, tag):
        assert tag == cls.full_tag(), 'Invalid tag, expected {} got {}'.format(cls.full_tag(), tag)


class TextLine(BaseElement):
    tag = 'TextLine'

    def __init__(self, id=None, coords=None, baseline=None, text_equiv=''):
        self.coords = coords if coords is not None else []  # type: List[Point]
        self.baseline = baseline if baseline is not None else []  # type: List[Point]
        self.id = id  # type: Optional[str]
        self.text_equiv = text_equiv  # type: str

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'TextLine':
        cls.check_tag(e.tag)
        return TextLine(
            id=e.attrib.get('id'),
            coords=Point.list_from_xml(e.find('p:Coords', _ns)),
            baseline=Point.list_from_xml(e.find('p:Baseline', _ns)),
            text_equiv=_get_text_equiv(e)
        )

    @classmethod
    def from_array(cls, cv2_coords: np.array=None, baseline_coords: np.array=None,  # shape [N, 1, 2]
                   text_equiv: str=None, id: str=None):
        return TextLine(
            id=id,
            coords=Point.cv2_to_point_list(cv2_coords) if cv2_coords is not None else [],
            baseline=Point.cv2_to_point_list(baseline_coords) if baseline_coords is not None else [],
            text_equiv=text_equiv
        )

    def to_xml(self):
        line_et = ET.Element('TextLine')
        line_et.set('id', self.id if self.id is not None else '')
        if not not self.coords:
            line_coords = ET.SubElement(line_et, 'Coords')
            line_coords.set('points', Point.list_point_to_string(self.coords))
        if not not self.baseline:
            line_baseline = ET.SubElement(line_et, 'Baseline')
            line_baseline.set('points', Point.list_point_to_string(self.baseline))
        line_text_equiv = ET.SubElement(line_et, 'TextEquiv')
        text_unicode = ET.SubElement(line_text_equiv, 'Unicode')
        if not not self.text_equiv:
            text_unicode.text = self.text_equiv
        return line_et

    def scale_baseline_points(self, ratio):
        scaled_points = list()
        for pt in self.baseline:
            scaled_points.append(Point(int(pt.y * ratio[0]), int(pt.x * ratio[1])))

        self.baseline = scaled_points


class GraphicRegion(BaseElement):
    tag = 'GraphicRegion'

    def __init__(self, id=None, coords=None,):
        self.coords = coords if coords is not None else []  # type: List[Point]
        self.id = id  # type: Optional[str]

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'GraphicRegion':
        cls.check_tag(e.tag)
        return GraphicRegion(
            id=e.attrib.get('id'),
            coords=Point.list_from_xml(e.find('p:Coords', _ns))
        )


class TextRegion(BaseElement):
    tag = 'TextRegion'

    def __init__(self, id=None, coords=None, text_lines=None, text_equiv=''):
        self.id = id  # type: Optional[str]
        self.coords = coords if coords is not None else []  # type: List[Point]
        self.text_equiv = text_equiv  # type: str
        self.text_lines = text_lines if text_lines is not None else []  # type: List[TextLine]

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'TextRegion':
        cls.check_tag(e.tag)
        return TextRegion(
            id=e.attrib.get('id'),
            coords=Point.list_from_xml(e.find('p:Coords', _ns)),
            text_lines=[TextLine.from_xml(tl) for tl in e.findall('p:TextLine', _ns)],
            text_equiv=_get_text_equiv(e)
        )

    def to_xml(self):
        text_et = ET.Element('TextRegion')
        text_et.set('id', self.id if self.id is not None else '')
        if not not self.coords:
            text_coords = ET.SubElement(text_et, 'Coords')
            text_coords.set('points', Point.list_point_to_string(self.coords))
        for tl in self.text_lines:
            text_et.append(tl.to_xml())
        text_equiv = ET.SubElement(text_et, 'TextEquiv')
        text_unicode = ET.SubElement(text_equiv, 'Unicode')
        # TODO : TextEquiv
        return text_et


class Page(BaseElement):
    tag = 'Page'

    def __init__(self, image_filename=None, image_width=None, image_height=None,
                 text_regions=None, graphic_regions=None):
        self.image_filename = image_filename  # type: Optional[str]
        self.image_width = _try_to_int(image_width)  # type: Optional[int]
        self.image_height = _try_to_int(image_height)  # type: Optional[int]
        self.text_regions = text_regions if text_regions is not None else []  # type: List[TextRegion]
        self.graphic_regions = graphic_regions if graphic_regions is not None else []  # type: List[GraphicRegion]

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'Page':
        cls.check_tag(e.tag)
        return Page(
            image_filename=e.attrib.get('imageFilename'),
            image_width=e.attrib.get('imageWidth'),
            image_height=e.attrib.get('imageHeight'),
            text_regions=[TextRegion.from_xml(tr) for tr in e.findall('p:TextRegion', _ns)],
            graphic_regions=[GraphicRegion.from_xml(tr) for tr in e.findall('p:GraphicRegion', _ns)]
        )

    @classmethod
    def from_dict(cls, dictionary: dict) -> 'Page':
        return Page(image_filename=dictionary.get('image_filename'),
                    image_width=dictionary.get('image_width'),
                    image_height=dictionary.get('image_height'),
                    text_regions=dictionary.get('text_regions'),
                    graphic_regions=dictionary.get('graphic_regions')
                    )

    def to_xml(self):
        page_et = ET.Element('Page')
        page_et.set('imageFilename', self.image_filename)
        page_et.set('imageWidth', str(self.image_width))
        page_et.set('imageHeight', str(self.image_height))
        for tr in self.text_regions:
            page_et.append(tr.to_xml())
        # TODO : complete graphic regions
        return page_et


def parse_file(filename: str) -> Page:
    xml_page = ET.parse(filename)
    page_elements = xml_page.findall('p:Page', _ns)
    # TODO can there be multiple pages in a single XML file?
    assert len(page_elements) == 1
    return Page.from_xml(page_elements[0])


def create_xml_page(dictionary: dict, creator_name='DocSeg') -> 'Page':
    page = Page.from_dict(dictionary)

    # Create xml
    root = ET.Element('PcGts')
    root.set('xmlns', _ns['p'])
    # Metadata
    generated_on = str(datetime.datetime.now())
    metadata = ET.SubElement(root, 'Metadata')
    creator = ET.SubElement(metadata, 'Creator')
    creator.text = creator_name
    created = ET.SubElement(metadata, 'Created')
    created.text = generated_on
    last_change = ET.SubElement(metadata, 'LastChange')
    last_change.text = generated_on

    root.append(page.to_xml())
    for k, v in _attribs.items():
        root.attrib[k] = v

    return root