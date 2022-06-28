# object_location_detection_using_tensorflow

The script generates a dataset by pasting randomly rotated tools from the tool folder into images in images folder,
and saving the correct bounding boxes of that tools in the labels.csv file. 

After that the CNN model learns to detect this bounding boxes based on the images.

Example result of the script:
(The blue square is the correct bounding box and the green one is predicted by the CNN model)
![tool_detection](https://user-images.githubusercontent.com/67229687/176241488-6d38c5b7-ac69-4f27-bb49-0cca02192d97.png)
