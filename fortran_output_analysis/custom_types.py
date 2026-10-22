import numpy as np
from typing import Annotated
from numpy.typing import NDArray

ArrFloat64 = Annotated[NDArray[np.float64], "1D array"]
ArrFloat64_2D = Annotated[NDArray[np.float64], "2D array"]
ArrFloat64_3D = Annotated[NDArray[np.float64], "3D array"]

ArrComplex128 = Annotated[NDArray[np.complex128], "1D array"]
ArrComplex128_2D = Annotated[NDArray[np.complex128], "2D array"]
ArrComplex128_3D = Annotated[NDArray[np.complex128], "3D array"]
