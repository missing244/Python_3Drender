from .. import np,abc,Union,Iterator,TypeOfInt,TypeOfNumber


class BaseMesh(metaclass=abc.ABCMeta) :
    
    __slots__ = ["location", "translation", "scale", "rotation", "dots", "index"] 
    
    def __init__(self) -> None :
        self.location = np.zeros(3)
        self.translation = np.zeros(3)
        self.scale = np.ones(3)
        self.rotation = np.array([1.0, 0.0, 0.0, 0.0])

    @property
    def translation_matrix(self) -> np.ndarray[np.float32] :
        return np.array( [
            [1, 0, 0, self.translation[0]],
            [0, 1, 0, self.translation[1]],
            [0, 0, 1, self.translation[2]],
            [0, 0, 0, 1]], dtype=np.float32)

    @property
    def scale_matrix(self) -> np.ndarray[np.float32] :
        return np.array( [
            [self.scale[0], 0, 0, 0],
            [0, self.scale[1], 0, 0],
            [0, 0, self.scale[2], 0],
            [0, 0, 0, 1]], dtype=np.float32)

    @property
    def rotation_matrix(self) -> np.ndarray[np.float32] :
        a = self.rotation[0]; b=self.rotation[1]; c=self.rotation[2]; d=self.rotation[3]
        return np.array( [
            [1-2*(c**2)-2*(d**2), 2*b*c-2*a*d, 2*a*c+2*b*d, 0],
            [2*b*c+2*a*d, 1-2*(b**2)-2*(d**2), 2*c*d-2*a*b, 0],
            [2*b*d-2*a*c, 2*a*b+2*c*d, 1-2*(b**2)-2*(c**2), 0],
            [0, 0, 0, 1]], dtype=np.float32)

    @abc.abstractmethod
    def __iter__(self) : pass


class LineMesh(BaseMesh, metaclass=abc.ABCMeta) :
    """
    # 直线组对象
    定义一个直线组对象
    
    ---------------------

    Mesh.dots 储存了所有的三维点信息

    Mesh.index 储存了两个索引为一组的直线信息
    """

    def __repr__(self) -> str:
        return "<%s Dots=%s>" % (self.__class__.__name__, len(self.dots))

    def __init__(self, dots:Iterator[Union[int,float]], index:Iterator[int]) -> None :
        super().__init__()
        if any((not isinstance(i, TypeOfNumber) for i in dots)) : raise TypeError("点必须为数字类型")
        if any((not isinstance(i, TypeOfInt) for i in index)) : raise TypeError("点索引必须为数字类型")
        if len(dots) % 3 != 0 : raise ValueError("存在不完整的三维点信息")
        if len(index) % 2 != 0 : raise ValueError("直线索引的数量必须为2的倍数")

        self.dots:np.ndarray[np.float32] = np.array([np.append(dots[i:i+3], 1) for i in range(0, len(dots), 3)], dtype=np.float32)
        self.index:np.ndarray[np.uint32] = np.array(index, dtype=np.uint32)

    def __setattr__(self, name: str, value) -> None:
        if not hasattr(self, name) : super().__setattr__(name, value)
        else : raise Exception("无法对 %s 属性进行重新赋值" % name)

    def __iter__(self) :
        translation = self.translation_matrix
        scale = self.scale_matrix
        rotate = self.rotation_matrix
        for i in range(3) : translation[i][3] += self.location[i]

        for i in range(0, len(self.index), 2) :
            yield np.array( [
                translation.dot( scale.dot( rotate.dot( self.dots[self.index[i]] ) ) ),
                translation.dot( scale.dot( rotate.dot( self.dots[self.index[i+1]] ) ) )
            ], dtype=np.float32)

class TriangleMesh(BaseMesh, metaclass=abc.ABCMeta) :
    """
    # 三角形网格对象
    定义一个三角形网格对象
    
    ---------------------

    Mesh.dots 储存了所有的三维点信息

    Mesh.index 储存了三个索引为一组的三角网信息
    """

    __slots__ = ["dots", "index"] 

    def __repr__(self) -> str:
        return "<%s Dots=%s>" % (self.__class__.__name__, len(self.dots))

    def __init__(self, dots:Iterator[Union[int,float]], index:Iterator[int]) -> None :
        super().__init__()
        if any((not isinstance(i, TypeOfNumber) for i in dots)) : raise TypeError("点必须为数字类型")
        if any((not isinstance(i, TypeOfInt) for i in index)) : raise TypeError("点索引必须为数字类型")
        if len(dots) % 3 != 0 : raise ValueError("存在不完整的三维点信息")
        if len(index) % 3 != 0 : raise ValueError("三角网索引的数量必须为3的倍数")

        self.dots:np.ndarray[np.float32] = np.array([np.append(dots[i:i+3], 1) for i in range(0, len(dots), 3)], dtype=np.float32)
        self.index:np.ndarray[np.uint32] = np.array(index, dtype=np.uint32)

    def __setattr__(self, name: str, value) -> None:
        if not hasattr(self, name) : super().__setattr__(name, value)
        else : raise Exception("无法对 %s 属性进行重新赋值" % name)

    def __iter__(self) :
        translation = self.translation_matrix
        scale = self.scale_matrix
        rotate = self.rotation_matrix
        for i in range(3) : translation[i][3] += self.location[i]

        for i in range(0, len(self.index), 3) :
            yield np.array( [
                translation.dot( scale.dot( rotate.dot( self.dots[self.index[i]] ) ) ),
                translation.dot( scale.dot( rotate.dot( self.dots[self.index[i+1]] ) ) ),
                translation.dot( scale.dot( rotate.dot( self.dots[self.index[i+2]] ) ) )
            ], dtype=np.float32)

