from . import np,Union,Graphics,TypeOfNumber
import OpenGL.GL as GL

TypeAliasRenderObject = Union[Graphics.Graphic3D.BaseGraphic3D]
TypeOfRenderObject = (Graphics.Graphic3D.BaseGraphic3D, )
__all__ = ["Scene", "Perspective_Camera"]

def iter_pairwise(List:list) :
    n = 0 ; length  = len(List) - 1
    while n < length :
        yield List[n]
        n += 1
        yield List[n]

def vector_cross(v1:np.ndarray[np.float32], v2:np.ndarray[np.float32]) :
    return np.array([
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0], 0], dtype=np.float32)




def Local_Tansfor(local_matrix:np.ndarray, camera_location:np.ndarray, dot:np.ndarray) -> np.ndarray[np.float32] :
    "将世界坐标点经过摄像机局部坐标系的变换\n\n获得基于摄像机局部坐标系的坐标点"
    new_dot = dot[0:3] - camera_location
    return np.array([
        np.dot(new_dot, local_matrix[0]), 
        np.dot(new_dot, local_matrix[1]), 
        np.dot(new_dot, local_matrix[2]), 1],
        dtype=np.float32
    )

def Camera_Projection(projection_matrix:np.ndarray, dot:np.ndarray) -> np.ndarray[np.float32] :
    "将坐标点经过摄像机投影矩阵的变换\n\n将坐标点变换为裁剪空间的坐标"
    projection_point = projection_matrix.dot( dot )
    return ((projection_point / projection_point[3]) if projection_point[3] else projection_point)[0:3]

def Front_Test(dot1:np.ndarray, dot2:np.ndarray, dot3:np.ndarray) -> bool :
    "用于测试三角形网格面是否为正面\n\n传入点的坐标应该为顺时针方向"
    crossproduct_vector = vector_cross(dot2 - dot1, dot3 - dot1)
    return np.dot(crossproduct_vector, dot1) < 0




Crop_NearPlane_Vector = np.array([0.0, 0.0, -1.0])

def Cropping_Test(dot1:np.ndarray[np.float32], dot2:np.ndarray[np.float32]) :
    """
    粗略判断直线是否与平面相交
    * 返回 False 代表直线在近平面的外侧(需忽略该直线)
    * 返回 True 代表直线在近平面的内侧(直接渲染该直线)
    * 返回 -1 代表直线与近平面相交(需另外计算交点并裁剪)
    """
    if dot1[2] >= 0 and dot2[2] >= 0 : return True
    elif dot1[2] < 0 and dot2[2] < 0 : return False
    else : return -1

def Line_Plane_Intersection(plane_point:np.ndarray[np.float32], plane_normal:np.ndarray[np.float32], 
    line_start:np.ndarray[np.float32], line_end:np.ndarray[np.float32]) :
    """
    计算直线与平面的交点
    * 返回 None 表示无交点
    * 返回 numpy.ndarray 表示交点坐标
    """
    line_start_point = line_start[0:3]
    line_end_point = line_end[0:3]

    # 确保向量都是单位向量
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    line_direction = (line_start_point - line_end_point) / np.linalg.norm(line_start_point - line_end_point)
    dot_product = np.dot(plane_normal, line_direction)
    if dot_product == 0 : 
        prove = sum( (i*(j-k) for i,j,k in zip(plane_normal, line_start_point, plane_point)) )
        if prove : return None
        else : return line_start

    # 计算从直线点到平面点的向量
    w = plane_point - line_start_point

    # 计算交点
    factor = np.dot(plane_normal, w) / dot_product
    intersection:np.ndarray[np.float32] = line_start_point + factor * line_direction

    return np.append(intersection, 1)

def Line_Cropping(near_plane_posz:float, dot1:np.ndarray, dot2:np.ndarray) : 
    preliminary_testing = Cropping_Test(dot1, dot2)

    if preliminary_testing == -1 :
        if dot1[2] > 0 : dot2, dot1 = dot1.copy(), dot2.copy()
        intersection_dot = Line_Plane_Intersection(np.array([0.0, 0.0, near_plane_posz]), Crop_NearPlane_Vector, dot1, dot2)
        if intersection_dot is None : return None, None
        return intersection_dot, dot2
    elif preliminary_testing : return dot1, dot2
    else: return None, None








class Scene :

    """储存每个3D图形对象的场景对象"""

    __slots__ = ["graphics"]

    def __init__(self, *graphs:TypeAliasRenderObject) -> None :
        if any( (not isinstance(i, TypeOfRenderObject) for i in graphs) ) : raise ValueError("无法添加非图形对象")
        self.graphics = {id(i):i for i in graphs}

    def __getitem__(self, index:int) -> TypeAliasRenderObject :
        key = tuple(self.graphics.keys())[index]
        return self.graphics[key]

    def __iter__(self) :
        for key in self.graphics : yield self.graphics[key]

    def add(self, graphic):
        """增加一个3D图形"""
        self.graphics[id(graphic)] = graphic
    
    def remove(self, graphic) :
        """移除一个3D图形"""
        del self.graphics[id[id(graphic)]]

class Perspective_Camera :
    
    """
    透视相机对象
    """
    __slots__ = ["scene","FOV","near_distance","far_distance",
                 "width","height","location","rotation"]

    def __setattr__(self, name: str, value) -> None:
        if not hasattr(self, name) :
            if name == "aspect_ratio" and not isinstance(value, TypeOfNumber) : raise TypeError("aspect_ratio 需要浮点数")
            elif name == "FOV" and not isinstance(value, TypeOfNumber) : raise TypeError("fov 需要浮点数")
            elif name == "scene" and not isinstance(value, Scene) : raise TypeError("scene 需要为场景对象")
            elif name == "near_distance" and not isinstance(value, TypeOfNumber) : raise TypeError("near 需要整数")
            elif name == "far_distance" and not isinstance(value, TypeOfNumber) : raise TypeError("far 需要整数")
            super().__setattr__(name, value)
        else :
            if not isinstance(getattr(self,name), (tuple, list, np.ndarray)) : raise TypeError("无法对 %s 属性赋值为非array类型" % name)
            elif getattr(self,name).__len__() != len(value) : raise ValueError("无法对 %s 属性赋值元素数量不同的array类型" % name)
            super().__setattr__(name, np.array(value, dtype=np.float32))


    def __init__(self, width:int, height:int, Scene:Scene, fov:float, near:float=0.1, far:float=500.0) -> None :
        if not isinstance(width, int) : raise TypeError("width 需要整数")
        if not isinstance(height, int) : raise TypeError("height 需要整数")

        self.scene = Scene
        self.FOV = fov
        self.near_distance = near
        self.far_distance = far

        self.width, self.height = width, height
        self.location = np.zeros(3)
        self.rotation = np.zeros(3)


    def __GLRender__(self, dots:np.ndarray[np.float32]) : 
        GL.glBegin(GL.GL_LINES)
        for dot in dots : GL.glVertex3d(*dot)
        GL.glEnd()

    def __calculate__(self, graphic:TypeAliasRenderObject, projection_matrix:np.ndarray, local_matrix:np.ndarray) :

        for gird_dots in graphic.mesh :
            local_dots = np.array( [Local_Tansfor(local_matrix, self.location, i) for i in gird_dots], dtype=np.float32)
            if isinstance(graphic.mesh, Graphics.Mesh.TriangleMesh) and \
                not Front_Test(local_dots[0], local_dots[1], local_dots[2]) : continue

            if not isinstance(graphic.mesh, Graphics.Mesh.LineMesh) : local_dots = np.array([*local_dots, local_dots[0]], dtype=np.float32)
            line_of_local_dots = np.array( [i for i in iter_pairwise(local_dots)], dtype=object)

            for index in range(0, len(line_of_local_dots), 2) :
                d1,d2 = Line_Cropping(self.near_distance, line_of_local_dots[index], line_of_local_dots[index+1])
                line_of_local_dots[index], line_of_local_dots[index+1] = d1,d2

            projection_dots = np.array( [Camera_Projection(projection_matrix, i) 
                for i in line_of_local_dots if i[0] is not None], dtype=np.float32)
            self.__GLRender__( projection_dots )

    def __call__(self) :
        projection_matrix = self.projection_matrix
        local_matrix = self.local_matrix
        for graph in self.scene : self.__calculate__(graph, projection_matrix, local_matrix)


    @property
    def projection_matrix(self) :
        fov_rad = 1 / np.tan(np.deg2rad(self.FOV * 0.5))
        m1 = self.far_distance / (self.far_distance - self.near_distance)

        matrix = np.zeros((4, 4))
        matrix[0][0] = (self.height / self.width) * fov_rad
        matrix[1][1] = fov_rad
        matrix[2][2] = m1
        matrix[3][2] = 1.0
        matrix[2][3] = -self.near_distance / m1
        return matrix

    @property
    def local_matrix(self) :
        a,b,c = self.rotation[0], self.rotation[1], self.rotation[2]
        RotateX = np.zeros((3, 3), dtype=np.float32)
        RotateX[0][0] = 1 ; RotateX[1][1] = np.cos(a) ; RotateX[1][2] = -np.sin(a)
        RotateX[2][1] = np.sin(a) ; RotateX[2][2] = np.cos(a)

        RotateY = np.zeros((3, 3), dtype=np.float32)
        RotateY[0][0] = np.cos(b) ; RotateY[0][2] = np.sin(b) ; RotateY[1][1] = 1
        RotateY[2][0] = -np.sin(b) ; RotateY[2][2] = np.cos(b)

        RotateZ = np.zeros((3, 3), dtype=np.float32)
        RotateZ[0][0] = np.cos(c) ; RotateZ[0][1] = -np.sin(c)
        RotateZ[1][0] = np.sin(c) ; RotateZ[1][1] = np.cos(c) ; RotateZ[2][2] = 1

        m1 = RotateZ.dot( RotateX.dot( RotateY ) )
        return m1
    
