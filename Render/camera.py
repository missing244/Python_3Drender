from . import np,Union,Tuple,Graphics,TypeOfNumber
import OpenGL.GL as GL

TypeAliasRenderObject = Union[Graphics.Graphic3D.BaseGraphic3D]
TypeOfRenderObject = (Graphics.Graphic3D.BaseGraphic3D, )
__all__ = ["Scene", "Perspective_Camera"]



def Local_Tansfor(local_matrix:np.ndarray, camera_location:np.ndarray, dot:np.ndarray) -> np.ndarray[np.float64] :
    "将世界坐标点经过摄像机局部坐标系的变换\n\n获得基于摄像机局部坐标系的坐标点"
    new_dot = dot - camera_location
    return np.array([np.dot(new_dot, local_matrix[0]), np.dot(new_dot, local_matrix[1]), np.dot(new_dot, local_matrix[2])])

def Camera_Projection(projection_matrix:np.ndarray, dot:np.ndarray) -> np.ndarray[np.float64] :
    "将坐标点经过摄像机投影矩阵的变换\n\n将坐标点变换为裁剪空间的坐标"
    projection_point = projection_matrix.dot( np.append(dot, 1) )
    return ((projection_point / projection_point[3]) if projection_point[3] else projection_point)[0:3]

def Front_Test(dot1:np.ndarray, dot2:np.ndarray, dot3:np.ndarray) -> bool :
    "用于测试三角形网格面是否为正面\n\n传入点的坐标应该为顺时针方向"
    crossproduct_vector = np.cross(dot2 - dot1, dot3 - dot1)
    return np.dot(crossproduct_vector, dot1) < 0




Crop_Space_Plane_Point = np.array([
    np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 
    np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]) ])
Crop_Space_Normal_Vector = np.array([
    np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 
    np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0,  0.0]) ])

def Cohen_Sutherland_Cropping_Test(dot1:np.ndarray, dot2:np.ndarray) :
    """
    粗略判断直线是否与裁剪空间相交
    https://blog.csdn.net/weixin_44397852/article/details/109015504
    https://www.cnblogs.com/fortunely/p/17739080.html
    * 返回 False 代表直线与裁剪空间无交点
    * 返回 True 代表直线完全位于裁剪空间内
    * 返回 Tuple 代表需要另外进行计算是否相交
    """

    encode_number = np.uint16(0)
    if dot1[1] > 1 :  encode_number |= 0b100000_000000
    if dot1[1] < -1 : encode_number |= 0b010000_000000
    if dot1[0] > 1 :  encode_number |= 0b001000_000000
    if dot1[0] < -1 : encode_number |= 0b000100_000000
    if dot1[2] > 1 :  encode_number |= 0b000010_000000
    if dot1[2] < -1 : encode_number |= 0b000001_000000
    if dot2[1] > 1 :  encode_number |= 0b100000
    if dot2[1] < -1 : encode_number |= 0b010000
    if dot2[0] > 1 :  encode_number |= 0b001000
    if dot2[0] < -1 : encode_number |= 0b000100
    if dot2[2] > 1 :  encode_number |= 0b000010
    if dot2[2] < -1 : encode_number |= 0b000001

    encode_dot1 = encode_number >> 6
    encode_dot2 = encode_number & 0b111111
    if encode_dot1 & encode_dot2 : return False
    if not(encode_dot1 | encode_dot2) : return True

    return encode_dot1, encode_dot2

def Line_Plane_Intersection(plane_point:np.ndarray[np.float64], plane_normal:np.ndarray[np.float64], 
    line_start_point:np.ndarray[np.float64], line_end_point:np.ndarray[np.float64]) :
    """
    计算直线与平面的交点
    * 返回 False 表示无交点
    * 返回 True 表示直线属于平面
    * 返回 numpy.ndarray 表示交点坐标
    """

    # 确保向量都是单位向量
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    line_direction = (line_start_point - line_end_point) / np.linalg.norm(line_start_point - line_end_point)
    dot_product = np.dot(plane_normal, line_direction)
    if dot_product == 0 : 
        prove = sum( (i*(j-k) for i,j,k in zip(plane_normal, line_start_point, plane_point)) )
        if prove : return False
        else : return True

    # 计算从直线点到平面点的向量
    w = plane_point - line_start_point

    # 计算交点
    factor = np.dot(plane_normal, w) / dot_product
    intersection:np.ndarray[np.float64] = line_start_point + factor * line_direction
    if any( ( (not min(i,k)<=j<=max(i,k)) for i,j,k in zip(line_start_point, intersection, line_end_point)) ) :
        return False

    return intersection

def Line_Cropping(dot1:np.ndarray, dot2:np.ndarray) : 
    preliminary_testing = Cohen_Sutherland_Cropping_Test(dot1, dot2)
    if preliminary_testing is False : return None, None
    elif preliminary_testing is True : return dot1, dot2

    for intersection_dots in (Line_Plane_Intersection(i, j, dot1, dot2) 
    for i,j in zip(Crop_Space_Plane_Point, Crop_Space_Normal_Vector)) :
        if intersection_dots is False : continue
        






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
            super().__setattr__(name, np.array(value, dtype=np.float64))


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


    def __GLRender__(self, *dots:np.ndarray[np.float64]) : 
        GL.glBegin(GL.GL_LINES)
        for index in range(len(dots) - 1) : 
            dot1, dot2 = Line_Cropping(dots[index], dots[index+1])
            if dot1 is None and dot2 is None : continue

            GL.glVertex3d(*dot1)
            GL.glVertex3d(*dot2)
        GL.glEnd()

    def __calculate__(self, graphic:TypeAliasRenderObject, projection_matrix:np.ndarray, local_matrix:np.ndarray) :

        for gird_dots in graphic.mesh :
            local_dots = np.array( [Local_Tansfor(local_matrix, self.location, i) for i in gird_dots], dtype=np.float64)
            projection_dots = np.array( [Camera_Projection(projection_matrix, i) for i in local_dots], dtype=np.float64)
            #print(local_dots, "\n" , projection_dots, "\n##########################")
            if isinstance(graphic.mesh, Graphics.Mesh.LineMesh) : 
                self.__GLRender__( *projection_dots )
            elif isinstance(graphic.mesh, Graphics.Mesh.TriangleMesh) and \
                Front_Test(local_dots[0], local_dots[1], local_dots[2]) : 
                self.__GLRender__( *projection_dots, projection_dots[0] )
            elif isinstance(graphic.mesh, Graphics.Mesh.QuadrilateralMesh) :
                if Front_Test(local_dots[0], local_dots[1], local_dots[2]) : 
                    self.__GLRender__( *projection_dots[0:3] )
                if Front_Test(local_dots[2], local_dots[3], local_dots[0]) : 
                    self.__GLRender__( *projection_dots[-2:], projection_dots[0] )
                #print(tansformation_dots)

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
        RotateX = np.array([
            [1.0, 0.0, 0.0], 
            [0.0, np.cos(a), np.sin(a)], 
            [0.0, -np.sin(a), np.cos(a)]])
        RotateY = np.array([
            [np.cos(b), 0.0, -np.sin(b)], 
            [0.0, 1.0, 0.0], 
            [np.sin(b), 0.0, np.cos(b)]])
        RotateZ = np.array([
            [np.cos(c), np.sin(c), 0.0], 
            [-np.sin(c), np.cos(c), 0.0], 
            [0.0, 0.0, 1.0]])
        
        m1 = RotateZ.dot( RotateX.dot( RotateY ) )
        return m1