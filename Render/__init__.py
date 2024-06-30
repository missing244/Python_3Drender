import numpy as np
import OpenGL.GLUT as GLUT
import OpenGL.GL as GL

from typing import Union,Iterator,Callable,Generator,List,Tuple
import abc

TypeOfInt = (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
TypeOfNumber = (int, float, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64)



def register_render_callback(width:int, height:int, fps:int, rendercall:Callable, windowbindcall:Callable=None) :
    GLUT.glutInitWindowSize(width, height)
    GLUT.glutInitDisplayMode(GLUT.GLUT_DOUBLE | GLUT.GLUT_RGBA)
    GLUT.glutInit()
    GLUT.glutCreateWindow("Render 3D")

    def render(value=0) :
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        rendercall()
        GLUT.glutTimerFunc(int(1000/fps), render, 0)
        GLUT.glutSwapBuffers()

    if hasattr(windowbindcall, "__call__") : windowbindcall()
    GLUT.glutDisplayFunc(rendercall)
    GLUT.glutTimerFunc(int(1000/fps), render, 0)
    GLUT.glutMainLoop()



from . import math as Math
from . import graphics as Graphics
from . import camera as Camera