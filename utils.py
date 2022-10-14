import cv2

def draw_bd_handpose(img,point):
    thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
    #
    cv2.line(img, (int(point[0][0]), int(point[0][1])),(int(point[1][0]), int(point[1][1])), colors[0], thick)
    cv2.line(img, (int(point[1][0]), int(point[1][1])), (int(point[2][0]), int(point[2][1])), colors[0], thick)
    cv2.line(img, (int(point[2][0]), int(point[2][1])), (int(point[3][0]), int(point[3][1])), colors[0], thick)
    cv2.line(img, (int(point[3][0]), int(point[3][1])), (int(point[4][0]), int(point[4][1])), colors[0], thick)

    cv2.line(img, (int(point[0][0]), int(point[0][1])), (int(point[5][0]), int(point[5][1])), colors[1], thick)
    cv2.line(img, (int(point[5][0]), int(point[5][1])), (int(point[6][0]), int(point[6][1])), colors[1], thick)
    cv2.line(img, (int(point[6][0]), int(point[6][1])), (int(point[7][0]), int(point[7][1])), colors[1], thick)
    cv2.line(img, (int(point[7][0]), int(point[7][1])), (int(point[8][0]), int(point[8][1])), colors[1], thick)

    cv2.line(img, (int(point[0][0]), int(point[0][1])), (int(point[9][0]), int(point[9][1])), colors[2], thick)
    cv2.line(img, (int(point[9][0]), int(point[9][1])), (int(point[10][0]), int(point[10][1])), colors[2], thick)
    cv2.line(img, (int(point[10][0]), int(point[10][1])), (int(point[11][0]), int(point[11][1])), colors[2], thick)
    cv2.line(img, (int(point[11][0]), int(point[11][1])), (int(point[12][0]), int(point[12][1])), colors[2], thick)

    cv2.line(img, (int(point[0][0]), int(point[0][1])), (int(point[13][0]), int(point[13][1])), colors[3], thick)
    cv2.line(img, (int(point[13][0]), int(point[13][1])), (int(point[14][0]), int(point[14][1])), colors[3], thick)
    cv2.line(img, (int(point[14][0]), int(point[14][1])), (int(point[15][0]), int(point[15][1])), colors[3], thick)
    cv2.line(img, (int(point[15][0]), int(point[15][1])), (int(point[16][0]), int(point[16][1])), colors[3], thick)

    cv2.line(img, (int(point[0][0]), int(point[0][1])), (int(point[17][0]), int(point[17][1])), colors[4], thick)
    cv2.line(img, (int(point[17][0]), int(point[17][1])), (int(point[18][0]), int(point[18][1])), colors[4], thick)
    cv2.line(img, (int(point[18][0]), int(point[18][1])), (int(point[19][0]), int(point[19][1])), colors[4], thick)
    cv2.line(img, (int(point[19][0]), int(point[19][1])), (int(point[20][0]), int(point[20][1])), colors[4], thick)
    return img