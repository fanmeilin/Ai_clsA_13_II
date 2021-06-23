import sys
sys.path.append('external_lib/mmclassification')

import cv2
import matplotlib.pyplot as plt
from lib.controller import Controller


# load regressor
model = Controller(checkpoint='model/latest.pth', config_file='model/densenet.py')

# load image
img = cv2.imread('test.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# find bboxes of bad revit
results = model.infer([img])[0]
for classname, score in zip(results['classnames'], results['scores']):
    print("{}: {:.3f}".format(classname, score), end=' ')
print()

# visualize result
plt.figure(figsize=(12,8))
plt.imshow(img)
plt.show()
