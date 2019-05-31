from __future__ import print_function
import cv2
import numpy as np
import argparse
import math
import os


class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))


class ARViewer(object):
    def __init__(self, template=None, obj_path=None, cap=None, MIN_MATCHES=None):
        self.cap = None
        self.MIN_MATCHES = MIN_MATCHES or 10
        self.camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        self.template_path = template
        self.template = None
        self.obj = None
        if obj_path:
            self.load_obj(obj_path)
        if self.template_path:
            self.load_template(self.template_path)
        self.match_tolerance = 0.15

    def run(self):
        if self.template is None:
            print('No template provided')
            return
        if not self.cap:
            self.cap = cv2.VideoCapture(0)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Unable to capture video")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            orb = cv2.ORB_create()
            frame_kp, frame_des = orb.detectAndCompute(frame, None)

            w, h = self.template.shape[::-1]
            temp_kp, temp_des = orb.detectAndCompute(self.template, None)
            homography = None
            matches = bf.match(temp_des, frame_des)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > self.MIN_MATCHES:
                temp_pts = np.float32([temp_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                frame_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                homography, mask = cv2.findHomography(temp_pts, frame_pts, cv2.RANSAC, 5.0)

                if True:  # if args.rectangle:
                    h, w = self.template.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, homography)
                    frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 2, cv2.LINE_AA)
                if homography is not None:
                    projection = self.projection_matrix(self.camera_parameters, homography)
                    frame = self.render(frame, self.obj, projection, self.template, False)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def load_template(self, template_path):
        if os.path.exists(template_path):
            self.template = cv2.imread(template_path, 0)

    def load_obj(self, obj_path):
        self.obj = OBJ(obj_path, swapyz=True)

    def projection_matrix(self, camera_parameters, homography):
        homography = homography * (-1)
        rot_and_trans = np.dot(np.linalg.inv(camera_parameters), homography)
        col_1 = rot_and_trans[:, 0]
        col_2 = rot_and_trans[:, 1]
        col_3 = rot_and_trans[:, 2]
        div = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1/div
        rot_2 = col_2/div
        trans = col_3/div
        c = rot_1 + rot_2
        p = np.cross(rot_1, rot_2)
        d = np.cross(c, p)
        rot_1 = np.dot(c/np.linalg.norm(c, 2) + d/np.linalg.norm(d, 2), 1/math.sqrt(2))
        rot_2 = np.dot(c/np.linalg.norm(c, 2) - d/np.linalg.norm(d, 2), 1/math.sqrt(2))
        rot_3 = np.cross(rot_1, rot_2)
        projection = np.stack((rot_1, rot_2, rot_3, trans)).T
        return np.dot(camera_parameters, projection)

    def hex_to_rgb(self, hex_color):
        """
        Helper function to convert hex strings to RGB
        """
        hex_color = hex_color.lstrip('#')
        h_len = len(hex_color)
        return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

    def render(self, img, obj, projection, model, color=False):
        vertices = obj.vertices
        scale_matrix = np.eye(3)*3
        h, w = model.shape

        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            points = np.array([[p[0] + w / 2, p[1] + h/2, p[2]] for p in points])
            dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
            imgpts = np.int32(dst)
            if color is False:
                cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
            else:
                color = hex_to_rgb(face[-1])
                color = color[::-1]
                cv2.fillConvexPoly(img, imgpts, color)
        return img


obj_path = '/Users/rmaffesoli/Documents/personal/openCV/marker.obj'
template_file = '/Users/rmaffesoli/Documents/personal/openCV/marker_card_512.jpg'
ARV = ARViewer(template_file, obj_path)
ARV.run()
