

class ImageAnnotation:
    def __init__(self, image_name: str, image_id: int, width: float, height: float):
        self.image_name = image_name
        self.image_id = image_id
        self.width = width
        self.height = height
        self.annotations = []  # each is [category_id, [x, y, w, h]] like [ 2, [539.08, 384.29, 377.02, 48.94]]

    def add_annotation(self, category_id: int, bbox: list):
        self.annotations.append([category_id, bbox])
        return None

    def width(self) -> float:
        return self.width

    def height(self) -> float:
        return self.height

    def annotations(self) -> list:
        return self.annotations

    def image_name(self) -> str:
        return self.image_name()


