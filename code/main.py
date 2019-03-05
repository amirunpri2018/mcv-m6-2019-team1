
import os

import utils as u


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    f = os.path.join(ROOT_DIR, 'annotation_pascal', '02_car', 'frame_086.xml')
    # u.create_aicity_xml(None, None)

    f = os.path.join(ROOT_DIR, 'datasets', 'AICity_data', 'train', 'S03', 'c010', 'Anotation_40secs_AICITY_S03_C010.xml')
    print u.get_bboxes_from_aicity(f)
    # u.xml_pascal_to_aicity(f, pretty_print=True)
