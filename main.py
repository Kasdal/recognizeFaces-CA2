import os
import cv2
from flask_socketio import SocketIO
import face_recognition as face_rec
from flask import Flask, render_template, Response
import numpy as np
import socket
import eventlet


