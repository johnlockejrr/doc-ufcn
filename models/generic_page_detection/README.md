# Page detection model

* Pre-trained with new U-FCN code on Horae-600 and READ-BAD
* 2 active learning iterations on HOME-Page (98/20 and 88/6 images)
* Without padding
* With AMP
* 768px
* Evaluation based on json

|                 | IoU   | F1    | AP@[.5] | AP@[.75] | AP@[.5,.95] |
| --------------- | ----- | ----- | ------- | -------- | ----------- |
| HOME-Page test  | 93.92 | 95.84 | 98.98   | 98.98    | 97.61       |
| Horae-Page test | 96.68 | 98.31 | 99.76   | 98.49    | 98.08       |
| Horae test-300  | 95.66 | 97.27 | 98.87   | 98.45    | 97.38       |
