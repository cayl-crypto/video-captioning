import cv2
import numpy as np
import os

## ADD Frames folder to file path before running code.

def video_to_frame(file_paths):

    path_to_save = 'C:\\Users\\pc\\PycharmProjects\\video-captioning\\Frames'
    # get file path for desired video and where to save frames locally
    for i in range(len(file_paths)):
        cap = cv2.VideoCapture('%s' %(file_paths[i]))

        current_frame = 1

        while (True):

            # capture each frame
            ret, frame = cap.read()

            # Save frame as a jpg file
            name = 'vid%s_frame' % (i+1) + str(current_frame) + '.jpg'
            print ('Creating: ' + name)
            cv2.imwrite(os.path.join(path_to_save, name), frame)

            # keep track of how many images you end up with
            current_frame += 1

            # stop loop when video ends
            if not ret:
                break

        # release capture
        cap.release()
    print('Done.')

import os

def get_filepaths(directory):

    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
    #print(file_paths)
    video_to_frame(file_paths)
    return file_paths  # Self-explanatory.




def main():
    # Run the above function and store its results in a variable.
    full_file_paths = get_filepaths("C:\\Users\\pc\\PycharmProjects\\video-captioning\\YouTubeClips")


if __name__ == "__main__":
    main()
