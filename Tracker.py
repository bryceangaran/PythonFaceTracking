import cv2
import numpy as np

import vtk
import vtk.vtkFiltersGeneralPython as filters

from webcam_server import WebcamServer


class ViewTracker:
    def __init__(
        self,
        webcam_server,
        face_cascade,
        distance_func,
        face_width,
        average=3
    ):
        self.webcam_server = webcam_server
        self.face_cascade = face_cascade
        self.distance_func = distance_func
        self.face_width = face_width

        self.average = average
        self.positions = np.zeros((average, 3))
        self.positions[:, 2] = 1
        self.index = 0

        self.last_face_pos = np.array((0, 0))

    def add_position(self, w_img, h_img, x, y, w_face, h_face):
        '''
        Given an image width and height and the x, y, width and height of a detected
        face's bounding box, calculate the position vector and add it to the
        internal list of positions.
        '''
        d = self.distance_func(w_face)
        cm_per_pixel = self.face_width / w_face

        # TODO: check why h_face is needed to center y_face
        # but w_face is not needed for x_face
        x_face = x - (w_img / 2)
        y_face = y + h_face - (h_img / 2)

        r = np.array((x_face * cm_per_pixel, y_face * cm_per_pixel, d))

        self.positions[self.index, :] = r
        self.index = (self.index + 1) % self.average

    def detect_faces(self, img):
        if img is not None and img.size > 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if not isinstance(faces, tuple):
                return faces
        return None

    def get_image_with_rectangles(self, img=None):
        if img is None:
            img = self.webcam_server.retrieve().copy()
        faces = self.detect_faces(img)
        if faces is None:
            return img
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0xff, 0), 2)
        return img

    def get_current_position(self, unit=True, img=None):
        if img is None:
            img = self.webcam_server.retrieve()
        faces = self.detect_faces(img)
        if faces is not None:
            main_face = None
            shortest_dist = 0.0
            current_face_pos = self.last_face_pos
            # Each face is (x, y, w, h).
            for face in faces:
                face_pos = np.array(face[:2])
                face_dist = np.linalg.norm(face_pos - self.last_face_pos)
                if main_face is None or face_dist < shortest_dist:
                    main_face = face
                    shortest_dist = face_dist
                    current_face_pos = face_pos
            self.last_face_pos = current_face_pos
            if main_face is not None:
                self.add_position(*img.shape[:2], *main_face)
                # Python 2 note
                # Double position expansion generates a syntax error in Python 2.
                # Comment the previous line and try the following instead.
                #  self.add_position(*(img.shape[:2] + main_face))

        # Return the averaged position for smooth transitions.
        r = np.average(self.positions, axis=0)
        if unit:
            r /= np.linalg.norm(r)
        return r

    def get_image_and_position(self, img=None, unit=False):
        img = self.webcam_server.retrieve()
        img2 = self.get_image_with_rectangles(img=img)
        pos = self.get_current_position(img=img, unit=unit)
        return img2, pos

    def get_current_rotation(self):
        r = self.get_current_position()
        r0 = np.array((0, 0, 1))
        if np.any(r != r0):
            return np.cross(r0, r)
        else:
            return None

    def get_vtk_transform(self):
        rot_axis = self.get_current_rotation()
        if rot_axis is None:
            return None

        rot_angle = np.arcsin(np.linalg.norm(rot_axis)) * 180 / np.pi

        transform = vtk.vtkTransform()
        transform.RotateWXYZ(rot_angle, rot_axis)
        return transform

    def apply_vtk_transform(self, obj):
        transform = self.get_vtk_transform()
        if transform is None:
            return None

        transformFilter = filters.vtkTransformFilter()
        transformFilter.SetInputData(obj)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        return transformFilter.GetOutput()
