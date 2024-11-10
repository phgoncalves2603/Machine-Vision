def get_box_center(box):
    x1,y1,x2,y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

def get_box_width(box):
    return box[2]-box[0]