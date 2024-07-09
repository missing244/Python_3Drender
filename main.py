import numpy as np
import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

from Render import register_render_callback
from Render import Camera, Graphics

import itertools, time, cProfile

WIDTH, HEIGHT = 16*70, 9*70
G0 = Graphics.Graphic3D.Cube().set_location([0,0,5])
G1 = Graphics.Graphic3D.BaseGraphic3D(Graphics.Mesh.TriangleMesh, [-1,0,0, 0,0,1, 1,0,0], [0,1,2]).set_location([0,0,5])
G2 = [Graphics.Graphic3D.Cube().set_location([i,0,j]) for i,j in itertools.product(range(0, 101, 5), range(0, 101, 5))]
Scene = Camera.Scene( *G2 )
Camera1 = Camera.Perspective_Camera(WIDTH, HEIGHT, Scene, 90.0, 0.1, 50)


def keyboard(key, x, y) :
    aaa = Camera1.local_matrix
    if key == b'a' : Camera1.location -= aaa[0][0:3] * 0.3
    if key == b'd' : Camera1.location += aaa[0][0:3] * 0.3
    if key == b'w' : Camera1.location += aaa[2][0:3] * 0.3
    if key == b's' : Camera1.location -= aaa[2][0:3] * 0.3
    if key == b' ' : Camera1.location[1] += 0.3
    if key == 112 : Camera1.location[1] -= 0.3
    print(key, Camera1.location)
    cProfile.run("flash()", sort=1)
def Motion(x, y) :
    if (WIDTH//2) == x and (HEIGHT//2) == y : return None
    Camera1.rotation[1] -= np.pi / 180 * (WIDTH//2 - x) * 0.2
    Camera1.rotation[0] -= np.pi / 180 * (HEIGHT//2 - y) * 0.2
    Camera1.rotation[0] = min(np.pi/2, Camera1.rotation[0])
    Camera1.rotation[0] = max(-np.pi/2, Camera1.rotation[0])
    GLUT.glutWarpPointer(WIDTH//2, HEIGHT//2)
    print([i/np.pi*180 for i in Camera1.rotation])
def windowscall() :
    GLUT.glutSetCursor(GLUT.GLUT_CURSOR_NONE)
    GLUT.glutSpecialFunc(keyboard)
    GLUT.glutKeyboardFunc(keyboard)
    GLUT.glutPassiveMotionFunc(Motion)


def flash() : 
    GL.glColor3f(1, 1, 1)
    Camera1()



register_render_callback(WIDTH, HEIGHT, 90, flash, windowscall) 