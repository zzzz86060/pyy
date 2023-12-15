import cv2
def imread_photo(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imread(filename, flags)
if __name__ == "__main__":
    img = imread_photo("image/5.png")
    cv2.imshow("img",img)
    cv2.waitKey(0)