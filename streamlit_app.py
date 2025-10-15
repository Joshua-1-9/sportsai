import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip
import tempfile
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def extract_pose_and_draw(video_path, out_path="out.mp4"):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w, h = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    frame_angles = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = res.pose_landmarks.landmark
            l_sh, r_sh = [lm[11].x * w, lm[11].y * h], [lm[12].x * w, lm[12].y * h]
            l_el, r_el = [lm[13].x * w, lm[13].y * h], [lm[14].x * w, lm[14].y * h]
            l_hip, r_hip = [lm[23].x * w, lm[23].y * h], [lm[24].x * w, lm[24].y * h]
            shoulder_rot = calc_angle(l_hip, l_sh, r_sh)
            arm_angle = calc_angle(l_sh, l_el, r_el)
            hip_align = calc_angle(l_sh, l_hip, r_hip)
            frame_angles.append([shoulder_rot, arm_angle, hip_align])
        out.write(frame)
    cap.release()
    out.release()
    pose.close()
    return np.array(frame_angles), fps, out_path


def align_sequences(seq1, seq2):
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    aligned1, aligned2 = [], []
    for i, j in path:
        aligned1.append(seq1[i])
        aligned2.append(seq2[j])
    return np.array(aligned1), np.array(aligned2)


def generate_feedback_text(angles_user, angles_pro):
    diffs = np.abs(angles_user - angles_pro)
    feedbacks = []
    for shoulder_diff, arm_diff, hip_diff in diffs:
        msg = []
        if shoulder_diff > 20:
            msg.append("Rotate shoulders more")
        elif shoulder_diff < 10:
            msg.append("Good shoulder rotation")
        if arm_diff > 20:
            msg.append("Widen arm angle")
        elif arm_diff < 10:
            msg.append("Solid arm position")
        if hip_diff > 15:
            msg.append("Rotate hips more")
        elif hip_diff < 10:
            msg.append("Good hip turn")
        feedbacks.append(", ".join(msg) if msg else "Nice form!")
    return feedbacks


def create_feedback_video(user_path, pro_path, feedback_texts):
    clip_user = VideoFileClip(user_path).resize(height=360)
    clip_pro = VideoFileClip(pro_path).resize(height=360)
    final_clip = clips_array([[clip_user, clip_pro]])
    fps = final_clip.fps or 30

    feedback_overlays = []
    duration = min(final_clip.duration, len(feedback_texts) / fps)
    for i in range(len(
