from .. import np,Union,Iterator
from . import Mesh


class BaseGraphic3D :

    __slots__ = ["mesh", "location", "translation", "scale", "rotation"]

    def __repr__(self) -> str:
        return "<Object %s>" % self.__class__.__name__

    def __init__(self, mesh:Union[Mesh.TriangleMesh, Mesh.LineMesh, Mesh.QuadrilateralMesh], 
                 dots:Iterator[Union[int,float]], dot_index:Iterator[int]) :
        self.mesh:Union[Mesh.TriangleMesh, Mesh.LineMesh, Mesh.QuadrilateralMesh] = mesh(dots, dot_index)
        self.location = self.mesh.location
        self.translation = self.mesh.translation
        self.scale = self.mesh.scale
        self.rotation = self.mesh.rotation

    def __setattr__(self, name: str, value) -> None:
        if not hasattr(self, name) : super().__setattr__(name, value) ; return None
        elif name in {"mesh"} : raise TypeError("无法修改 %s 属性" % name)
        elif not isinstance(getattr(self,name), (list, np.ndarray)) : raise TypeError("无法对 %s 属性赋值为非array类型" % name)
        elif getattr(self,name).__len__() != len(value) : raise ValueError("无法对 %s 属性赋值元素数量不同的array类型" % name)
        for i in range(len(value)) : getattr(self, name)[i] = value[i]


    def set_location(self, pos:np.ndarray) :
        self.location = np.array(pos, dtype=np.float64)
        return self


class Cube(BaseGraphic3D) :
    
    def __init__(self) -> None :
        super().__init__(
            Mesh.TriangleMesh,
            [-0.5,-0.5,-0.5, -0.5,-0.5,0.5, 0.5,-0.5,0.5, 0.5,-0.5,-0.5, 
             -0.5,0.5,-0.5,  -0.5,0.5,0.5,  0.5,0.5,0.5,  0.5,0.5,-0.5],
            [0,4,7, 0,7,3, #南
             3,7,6, 3,6,2, #东
             2,6,5, 2,5,1, #北
             1,5,4, 1,4,0, #西
             4,5,6, 4,6,7, #上
             2,1,0, 2,0,3] #下  
        )


class Line(BaseGraphic3D) :

    def __init__(self) -> None :
        super().__init__(
            Mesh.LineMesh,
            [0,0.0,0.0, 1,0.0,0.0, 0.0,1.0,0.0, 0,0.0,1.0],
            [0,1, 0,2, 0,3]
        )
        self.a = [0,0,0]

    def __call__(self) :
        a,b,c = self.a[0], self.a[1], self.a[2]
        RotateX = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), np.sin(a)], [0.0, -np.sin(a), np.cos(a)]])
        RotateY = np.array([[np.cos(b), 0.0, -np.sin(b)], [0.0, 1.0, 0.0], [np.sin(b), 0.0, np.cos(b)]])
        RotateZ = np.array([[np.cos(c), np.sin(c), 0.0], [-np.sin(c), np.cos(c), 0.0], [0.0, 0.0, 1.0]])
        self.mesh.dots[1] = RotateZ.dot( RotateY.dot( RotateX.dot([1,0.0,0.0]) ) )
        self.mesh.dots[2] = RotateZ.dot( RotateY.dot( RotateX.dot([0,1.0,0.0]) ) )
        self.mesh.dots[3] = RotateZ.dot( RotateY.dot( RotateX.dot([0,0.0,1.0]) ) )
        self.a[2] += 1 * np.pi / 180 ; #print(self.a[0] / np.pi * 180)
        #print(self.mesh.dots)