"""
Provide full LBPH recognizer functionality for video stream by video camera.
"""
import os

import keyboard as keyboard
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyautogui as pyautogui

from constants import (
    default_frontal_face_cascade_path,
    default_right_eye_cascade_path,
    default_left_eye_cascade_path,
    default_model_backup_path,
    default_members_list_file,
    default_images_dataset_directory,
)


class LBPHModel:
    """
    DVF serializer implementation.
    """
    def __init__(
        self,
        frontal_face_cascade_path=default_frontal_face_cascade_path,
        right_eye_cascade_path=default_right_eye_cascade_path,
        left_eye_cascade_path=default_left_eye_cascade_path,
        lbph_model_parameters=None,
        model_backup_path=default_model_backup_path,
        members_list_file=default_members_list_file,
        images_dataset_directory=default_images_dataset_directory,
    ) -> None:
        # Initializing paths to different files and directories
        self.members_list_file = members_list_file
        self.images_dataset_directory = images_dataset_directory
        self.model_backup_path = model_backup_path

        # Initialization of classifiers according to the Viola-Jones algorithm(Haar cascades)
        self._frontal_face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)
        self._right_eye_cascade = cv2.CascadeClassifier(right_eye_cascade_path)
        self._left_eye_cascade = cv2.CascadeClassifier(left_eye_cascade_path)

        # Creating and restoring process for LBPH Model
        self.is_trained = False
        self.create(lbph_model_parameters)

        # Initialization members info
        self.members = {}
        self.actual_members_id = 0
        self.get_actual_members_info()

    def create(self, lbph_model_parameters):
        """
        Create LBPH model.
        If a backup file exists, the model will restore the previous training steps.

        Args:
            lbph_model_parameters: variables for LBPH algorithm(such as radius, neighbour, etc).
        """
        print('\n[INFO] Creation process for LBPH Model...')
        if lbph_model_parameters:
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                    lbph_model_parameters[0],
                    lbph_model_parameters[1],
                    lbph_model_parameters[2],
                    lbph_model_parameters[3],
                )
            except Exception as error:
                print(f"[ERROR] Something went wrong. Error: {error}")
        else:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        print('\n[INFO] LBPH Model has been created successfully.')

        if self.members_list_file and os.path.exists(self.members_list_file):
            self.load()
            self.is_trained = True
        else:
            print("[WARNING] LBPH model backup file does not exist or the path is incorrect.")

    def load(self):
        """
        Restore a previously trained LBPH model using a special backup file.
        """
        print('\n[INFO] Restoring an LBPH model using a backup file...')
        try:
            self.recognizer.read(self.model_backup_path)
        except Exception as error:
            print(f"[ERROR] Something went wrong. Error: {error}")
        print('\n[INFO] LBPH Model has been restored successfully.')

    def save(self):
        """
        Saving the actual trained LBPH model.
        """
        print('\n[INFO] LBPH Model saving process...')
        self.recognizer.write(self.model_backup_path)
        print('\n[INFO] LBPH Model has been successfully saved to a backup file.')

    def update(self, images_of_faces, labels_of_faces):
        """
        Update existing LBPH model with new member`s images of face.
        After updating the model, it is saved.

        Args:
            images_of_faces: new member`s images of face.
            labels_of_faces: id of new member.
        """
        print('\n[INFO] LBPH Model updating process for a new member...')
        self.recognizer.update(images_of_faces, labels_of_faces)
        print('\n[INFO] LBPH Model has been successfully updated for a new member.')
        self.save()

    def get_actual_members_info(self):
        """
        Updating the current information about members from a special file.
        """
        if os.path.exists(self.members_list_file):
            members = {}

            with open(self.members_list_file, "r") as file_reader:
                for line in file_reader:
                    member_id, member_name = line.split(',')
                    members.update({int(member_id): member_name})

            if members:
                self.members = members
                self.actual_members_id = max(members.keys())

    def new_member_dataset(self, images_of_faces):
        """
        Creating and saving a special picture with all taken face images on one general picture.
        """
        figure, axes = plt.subplots(10, 5, figsize=(20, 20), facecolor='w', edgecolor='k')
        figure.subplots_adjust(hspace=0.5, wspace=.001)

        for face_number, detected_face in enumerate(images_of_faces):
            axes[int(face_number / 5)][face_number % 5].imshow(detected_face, cmap='gray', vmin=0, vmax=255)
            axes[int(face_number / 5)][face_number % 5].set_title(
                f'{self.actual_members_id}_member_part_{face_number}.jpg',
                fontdict={'fontsize': 16, 'fontweight': 'medium'},
            )
            axes[int(face_number / 5)][face_number % 5].axis('off')

        plt.savefig(f'{self.actual_members_id}_member_datasets.jpg')

    def add_new_member(self):
        """
        The complex process of adding a new member, consisting of these steps:
            1) Entering the name of a new member
            2) Saving face images of the new member
            3) Updating the model
            4) Updating of the member data
            5) Creating a folder with the new member's ID and uploading photos to it
        """
        print('\n[INFO] LBPH Model updating process for a new member...')
        while True:
            new_member_name = input('\nPlease enter a name of new member and press <Enter> button: ')
            if new_member_name != '':
                break
            else:
                print('Empty string is not supporting!')

        detected_faces = self.face_detection()
        self.actual_members_id += 1
        labels_of_faces = [self.actual_members_id] * len(detected_faces)

        self.update(detected_faces, np.array(labels_of_faces))
        self.members.update({self.actual_members_id: new_member_name})
        self.is_trained = True

        with open(self.members_list_file, "a+") as file_writer:
            file_writer.write(str(self.actual_members_id) + ',' + new_member_name + '\n')

        os.makedirs(f'{self.images_dataset_directory}/{self.actual_members_id}')
        for counter in range(len(detected_faces)):
            cv2.imwrite(
                f'{self.images_dataset_directory}/{self.actual_members_id}/member_part_{counter}.jpg',
                detected_faces[counter]
            )

        self.new_member_dataset(detected_faces)

        print(f'\n[INFO] New member {new_member_name} with an actual id - {self.actual_members_id}'
              f' has been successfully added')

    def face_detection(self, images_limit: int = 50):
        """
        The main process of collecting images of a new member's face.

        Args:
           images_limit(by default=50): amount of face images.

        Returns:
           Valid face images(grayed out, rotated).
        """
        camera = cv2.VideoCapture(0)
        camera.set(3, 640)
        camera.set(4, 480)

        # Initialization of result images with new member`s face
        face_imageset = []
        frame_counter = 0

        while True:
            # Reading a single frame from the video stream
            ret, frame = camera.read()

            # Converting a color image to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display of each frame, for easy use of the program and smoother usage
            cv2.imshow("Video", frame)
            # Searching for a face in the gray image
            faces = self._frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

            # Check if there is only one face for correct learning process
            if len(faces) == 1:
                try:
                    face_x, face_y, face_width, face_height = faces[0]

                    # Selecting a face in the overall image
                    gray_face = gray[face_y: face_y + face_height, face_x: face_x + face_width]
                    # Selecting the upper left part of the face to find the right eye
                    gray_upper_left = gray_face[0:int(gray_face.shape[0] / 2), 0:int(gray_face.shape[1] / 2)]
                    # Selecting the upper right part of the face to find the left eye
                    gray_upper_right = gray_face[0:int(gray_face.shape[0] / 2), int(gray_face.shape[1] / 2):int(gray_face.shape[1])]

                    # Searching for a right eye on the face image
                    right_eye = self._right_eye_cascade.detectMultiScale(
                        gray_upper_left,
                        scaleFactor=1.05,
                        minNeighbors=6,
                        minSize=(10, 10)
                    )
                    # Check if there is only one right eye
                    if len(right_eye) == 1:
                        right_eye_x, right_eye_y, right_eye_width, right_eye_height = right_eye[0]

                        # Searching for a left eye on the face image
                        left_eye = self._left_eye_cascade.detectMultiScale(
                            gray_upper_right,
                            scaleFactor=1.05,
                            minNeighbors=6,
                            minSize=(10, 10)
                        )

                        # Check if there is only one left eye
                        if len(left_eye) == 1:
                            left_eye_x, left_eye_y, left_eye_width, left_eye_height = left_eye[0]

                            # Calculation of the angle between the eyes into rad by default
                            eyeXdis = (left_eye_x + face_width / 2 + left_eye_width / 2) - (right_eye_x + right_eye_width / 2)
                            eyeYdis = (left_eye_y + left_eye_height / 2) - (right_eye_y + right_eye_height / 2)
                            angle_rad = np.arctan(eyeYdis / eyeXdis)
                            # Convert rad to degree
                            angle_degree = angle_rad * 180 / np.pi

                            # Find the center of the image
                            image_center = tuple(np.array(gray_face.shape) / 2)
                            # Create rotation matrix with calculated angle between the eyes
                            rot_mat = cv2.getRotationMatrix2D(image_center, angle_degree, 1.0)
                            # Image rotation process
                            rotated_image = cv2.warpAffine(gray_face, rot_mat, gray_face.shape, flags=cv2.INTER_LINEAR)

                            # Adding a new valid face image to the new member's dataset
                            face_imageset.append(rotated_image)
                            frame_counter += 1
                            # cv2.imshow("gray_face", gray_face)
                            # cv2.imshow("rotated_image", rotated_image)

                            # Framing a face in a single frame from a streaming video
                            cv2.rectangle(frame, (face_x, face_y), (face_x + face_width, face_y + face_height), (255, 255, 255), 1)
                            cv2.imshow("Video", frame)

                except Exception as error:
                    print(f"[ERROR] Something went wrong. Error: {error}")
                    continue

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if frame_counter == images_limit:
                break

        camera.release()
        cv2.destroyAllWindows()

        return face_imageset

    def recognize(self):
        """
        The main process of recognizing member by images of a new member's face.
        If the model has not been trained before, the corresponding error will appear.
        And the recognizing process will be stopped.
        """
        if not self.is_trained:
            print("[WARNING] LBPH model is not trained, start by training at least one new member.")
            return self

        camera = cv2.VideoCapture(0)
        camera.set(3, 640)
        camera.set(4, 480)

        while True:
            # Reading a single frame from the video stream
            ret, frame = camera.read()

            # Converting a color image to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display of each frame, for easy use of the program and smoother usage
            cv2.imshow("Video", frame)
            # Searching for a face in the gray image
            faces = self._frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

            # check if there is at least one face on the frame of the camera
            if len(faces) > 0:
                try:
                    for face in faces:
                        face_x, face_y, face_width, face_height = face

                        # Selecting a face in the overall image
                        gray_face = gray[face_y: face_y + face_height, face_x: face_x + face_width]
                        # Selecting the upper left part of the face to find the right eye
                        gray_upper_left = gray_face[0:int(gray_face.shape[0]/2), 0:int(gray_face.shape[1]/2)]
                        # Selecting the upper right part of the face to find the left eye
                        gray_upper_right = gray_face[0:int(gray_face.shape[0]/2), int(gray_face.shape[1]/2):int(gray_face.shape[1])]

                        # Searching for a right eye on the face image
                        right_eye = self._right_eye_cascade.detectMultiScale(
                            gray_upper_left,
                            scaleFactor=1.05,
                            minNeighbors=6,
                            minSize=(10, 10)
                        )
                        # Check if there is only one right eye
                        if len(right_eye) == 1:
                            right_eye_x, right_eye_y, right_eye_width, right_eye_height = right_eye[0]

                            # Searching for a left eye on the face image
                            left_eye = self._left_eye_cascade.detectMultiScale(
                                gray_upper_right,
                                scaleFactor=1.05,
                                minNeighbors=6,
                                minSize=(10, 10)
                            )

                            # Check if there is only one left eye
                            if len(left_eye) == 1:
                                left_eye_x, left_eye_y, left_eye_width, left_eye_height = left_eye[0]

                                # Calculation of the angle between the eyes into rad by default
                                eyeXdis = (left_eye_x + face_width / 2 + left_eye_width / 2) - (right_eye_x + right_eye_width / 2)
                                eyeYdis = (left_eye_y + left_eye_height / 2) - (right_eye_y + right_eye_height / 2)
                                angle_rad = np.arctan(eyeYdis / eyeXdis)
                                # Convert rad to degree
                                angle_degree = angle_rad * 180 / np.pi

                                # Find the center of the image
                                image_center = tuple(np.array(gray_face.shape) / 2)
                                # Create rotation matrix with calculated angle between the eyes
                                rot_mat = cv2.getRotationMatrix2D(image_center, angle_degree, 1.0)
                                # Image rotation process
                                rotated_image = cv2.warpAffine(gray_face, rot_mat, gray_face.shape, flags=cv2.INTER_LINEAR)

                                # Prediction process
                                label_id, conf = self.recognizer.predict(rotated_image)
                                print(label_id, conf, f'{label_id} member on the frame with {conf} confidence value')
                                # Check that the face is recognized
                                if conf > 15:
                                    cv2.putText(frame, self.members[label_id], (face_x, int(face_y+face_height*1.1)), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255))
                                else:
                                    cv2.putText(frame, 'unknown', (face_x, int(face_y+face_height*1.1)), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255))

                                # Framing a face in a single frame from a streaming video
                                cv2.rectangle(frame, (face_x, face_y), (face_x + face_width, face_y + face_height), (255, 255, 255), 1)
                                cv2.imshow("Video", frame)

                                # cv2.imshow("gray_face", gray_face)
                                # cv2.imshow("rotated_image", rotated_image)

                except Exception as error:
                    print(f"[ERROR] Something went wrong. Error: {error}")
                    continue

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Main variables for LBPH algorithm
    radius = 1
    neighbour = 8
    grid_x = 8
    grid_y = 8

    # LBPH model instance
    main_model = LBPHModel(lbph_model_parameters=[radius, neighbour, grid_x, grid_y])

    while True:
        print('\nPlease, select one of the options:\n1) Add new member\n2) Recognize\n3) Exit\n')
        event = keyboard.read_event()

        if event.name == '1':
            print("You selected the 'Add new member' option\n")
            pyautogui.click()
            # Adding new member into model
            main_model.add_new_member()

        elif event.name == '2':
            print("You selected the 'Recognize' option\n")
            pyautogui.click()
            # Trying to predict who is in front of the camera
            main_model.recognize()

        elif event.name == '3' or event.name == 'q' or event.name == 'esc':
            print("You selected the 'Exit' option\nBye\n")
            pyautogui.click()
            exit()
