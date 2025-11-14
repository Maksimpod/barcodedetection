import cv2
import FreeSimpleGUI as sg
import numpy as np

sg.theme("Default1")

layout = [
  [sg.Text('Изображение'), sg.InputText(k="image"), sg.FileBrowse("Обзор")
   ],
  [sg.Text("Вывод: ")],
  [sg.Multiline(size=(88, 20), k="OUTPUT", autoscroll=True)],
  [sg.Button("Начать", k="Start")]
]

interface = sg.Window("Распознавание штрихкодов", layout)


def detect_barcode(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
  gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

  gradient = cv2.subtract(gradX, gradY)
  gradient = cv2.convertScaleAbs(gradient)

  blurred = cv2.blur(gradient, (3, 3))
  (_, thresh) = cv2.threshold(blurred, 210, 250, cv2.THRESH_BINARY)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
  closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

  closed = cv2.erode(closed, None, iterations=7)
  closed = cv2.dilate(closed, None, iterations=2)

  (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

  c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
  rect = cv2.minAreaRect(c)
  initial_box = np.intp(cv2.boxPoints(rect))

  x, y, w, h = cv2.boundingRect(initial_box)

  barcode = img[y:y + h, x:x + w]

  box = np.array([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h]
  ])[np.newaxis, :, :]

  return initial_box[np.newaxis, :, :], box, barcode


def decode(values):
  img = cv2.imread(values["image"])

  box, corners, barcode = detect_barcode(img)

  decoder = cv2.barcode.BarcodeDetector()
  retval, flag = decoder.decode(barcode, corners)

  if retval:
    interface["OUTPUT"].update(f"Найденный штрих-код: {retval} \n", append=True)

    if box is not None:
      points = box.astype(int)

      cv2.polylines(img, [points], True, (0, 255, 0), 2)

      text_x = int(points[0][0][0])
      text_y = int(points[0][1][1])

      cv2.putText(img, retval, (text_x, text_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("result", img)
    cv2.waitKey()
  else:
    interface["OUTPUT"].update(f"Штрих-код не обнаружен \n", append=True)


while True:
  event, values = interface.read()
  if event in (None, 'Exit', 'Cancel'):
    break
  if event == "Start":
    interface.start_thread(lambda: interface(decode(values)), "Stop")
