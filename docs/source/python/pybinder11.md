# Pybind11实现Python与C++无缝连接

## Pybind11用途

​	主要用于C++和python之间的相互调用是一轻量级仅标头库，语法类似Boost.Python使用编译时自动推断类型剥离了与绑定生成无关的所有内容，超越Boost.Python，简化了绑定代码。



## Windows 安装与设置


&emsp;&emsp;1. Windows系统

&emsp;&emsp;Microsoft Visaul Studio 2017 x64，Anaconda3 , with python 3.8 

&emsp;&emsp;2. Pybind11安装

&emsp;&emsp;pybind11是 header-only的，因此不需要编译动态链接库，直接使用即可。

&emsp;&emsp;1）下载pybind11 https://github.com/pybind/pybind11解压到C:\pybind11-master；	

&emsp;&emsp;2）环境变量添加： 

``` 
  C:\ProgramData\Anaconda3;
  C:\ProgramData\Anaconda3\Library\mingw-w64\bin;
  C:\ProgramData\Anaconda3\Library\usr\bin;
  C:\ProgramData\Anaconda3\Library\bin;
  C:\ProgramData\Anaconda3\Scripts;
  C:\ProgramData\Anaconda3\Lib\site-packages;
  C:\ProgramData\Anaconda3\Lib\site-packages\numpy\core\include;
  C:\ProgramData\Anaconda3\pkgs\mkl-2020.0-166\Library\bin;
  C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64;
  C:\pybind11-master\include;  
```

&emsp;&emsp;3）windows下python环境变量设置

      Admin用户变量：`PYTHONHOME  C:\ProgramData\Anaconda3`
                                   ` PYTHONPATH  C:\ProgramData\Anaconda3\Lib\site-packages `



&emsp;&emsp;3.  使用VS2017测试

&emsp;&emsp; 使用C++编写python扩展（python调用C++）新建一个vs c++工程 example

&emsp;&emsp; 工程配置：

&emsp;&emsp; 1）设置编译输出类型 

&emsp;&emsp;配置属性--常规--常规--目标文件扩展名：.pyd

&emsp;&emsp;配置属性--常规--项目默认值-配置类型：动态库.dll

&emsp;&emsp; 2）添加include包含： 
 	  
&emsp;&emsp;配置属性--VC++目录--常规--包含目录：`C:\pybind11-master\include`,`C:\ProgramData\Anaconda3\include`
 	
&emsp;&emsp; 3）链接器配置：

&emsp;&emsp;链接器-常规-附加库目录：`C:\ProgramData\Anaconda3\libs`; 链接器-输入-附加依赖项：`python3.lib`

&emsp;&emsp;文件内容：

```
//example1.h
		#pragma once
		int sub(int, int);
```

```
//example1.cpp
		#include "example1.h"
 
		int sub(int x, int y) { return x - y; }
```

```
//example.cpp
		#include <pybind11/pybind11.h>
		#include "example1.h"
 
		namespace py = pybind11;
 
		int add(int i, int j) {return i + j;}
 
		PYBIND11_MODULE(example, m) {
			m.doc() = "pybind11 example plugin"; // optional module docstring
			m.def("add", &add, "A function which adds two numbers");
			m.def("sub", &sub, "A function which adds two numbers");
		}
```



### 测试

在`example.pyd`目录下，执行以下python语句。

`>>import example as ex`

`>>ex.add(2,3)`





## Linux安装与配置

1. python安装： `pip install pybind11`

2) 编译cpp文件：`c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example.so`

   如果使用IDE编写cpp文件，可以把`python3 -m pybind11 --includes`输出的路径添加到include路径，这样IDE就能智能代码提示。

3) python测试：

   ```
   >>>import example as ex
   >>>ex.add(2,3)
   ```

   

## C++与Python交互

### Python类型

1. 有效的包裹类(Available wrappers)

   所有的Python类型都使用C++包裹类进行包裹。这些类可以直接当作函数的参数。有效的类型有，

   `handle, object, bool_, int_, float_, str, bytes, tuple, list, dict, slice, none, capsule, iterable, iterator, function, buffer, array, array_t`

2. 示例

   Python的字典类型可以在`dict`构造函数中进行初使化。

   ```c++
   using namesapce pybind11::literatls; //to bring in the  '_a' literal
   py::dict d("spam"_a=py::none(),"eggs"_a=42);
   ```

   Python的元组对像可以命名用`py::make_tuple()`来实例化。

   ```
   py::tuple tup=py::make_tuple(42,py::none(),"spame")
   ```

3. 来回转换

   使用`py::cast()`将C++类型转换为Python类型。

   ```c++
   MyClass *cls=...;
   py::object obj=py::cast(cls)
   ```

   反向转换如下，

   ```c++
   py::object obj=...;
   MyClass *cls=obj.cast<MyClass *>();
   ```

4. C++访问Python库

   ```c++
   // Equivalent to "from decial import Decimal"
   py::object Decimal=py::module_::import("decimal").attr("Decimal");
   // Try to import scipy
   py::object scipy=py::module_::import("scipy");
   return scipy.attr("__version__");
   ```

5. C++调用Python函数

   ```c++
   // Construct a Python object of class Decimal
   py::object pi=Decimal("3.14159");
   // Use Python to make our directories
   py::object os=py::module_::import("os");
   py::object makedirs = os.attr("makedirs");
   makedirs("/tmp/path/to/somewhere");
   ```

   如果类型转换或`py::class_`已定义，则可以将Python获得的计算结果转换为C++类型。

   ```c++
   py::function f=...;
   py::object result_py=f(123,"bac",some_instance);
   MyClass &result=result_py.cast<MyClass>();	
   ```

6. 参数关键字

   如下Python的调用语法，

   ```python
   def f(number, say, to):
   	... # function code
   f(123, say="hello",to=some_instance) # keyword call in Python
   ```

   在C++中，可以通过以下代码等效调用，

   ```c++
   using namespace pybind11::literals;  //to bring in the  _a literal
   f(123, "say"_a="hello","to"_a=some_instance); //keyword call in C++
   ```

7. 参数拆箱

   `*args, **kwargs`参数拆开同样是可行的，并且可以和其它参数混合。

   ```c++
   // * unpacking
   py::tuple args=py::make_tuple(1234, "hello", some_instance);
   f(*args);
   
   // ** unpacking
   py::dict kwargs = py::dict("number"_a=1234, "say"_a="hello", "to"_a=some_instance);
   f(**kwargs);
   
   // mixed keywords, * and ** unpacking
   py::tuple args = py::make_tuple(1234);
   py::dict kwargs = py::dict("to"_a=some_instance);
   f(*args, "say"_a="hello", **kwargs);
   ```

8. 隐式转换

   当使用Python类型的C++接口，或调用Python函数，一般返回的是`object`类型的对象。可以隐式转换为子类型，如`dict`。这与代理对象的`operator[]`或`obj.attr()`返回结果是等效的。

   ```c++
   #include <pybind11/numpy.h>
   using namespace pybind11::literals;
   
   py::module_ os = py::module_::import("os");
   py::module_ path = py::module_::import("os.path");  // like 'import os.path as path'
   py::module_ np = py::module_::import("numpy");  // like 'import numpy as np'
   
   py::str curdir_abs = path.attr("abspath")(path.attr("curdir"));
   py::print(py::str("Current directory: ") + curdir_abs);
   py::dict environ = os.attr("environ");
   py::print(environ["HOME"]);
   py::array_t<float> arr = np.attr("ones")(3, "dtype"_a="float32");
   py::print(py::repr(arr + py::int_(1)));
   ```

   



### Numpy

1. Python缓冲协议buffer_protocol

   Python 中可用的某些对象封装了对底层内存数组或缓冲区的访问。这类对象包含内建的`bytes`和`bytearray`以及一些扩展的类型如`array.array`。第三方库为了某些目标，或许也定义了它们自有类型，例如图像处理或数值分析。尽管这些类型都有它们自已的语法，但都一个共同的特征，即背后可能都有一个非常大的内存缓冲区。在某些情况下，希望直接访问缓冲区，而无需中间复制。Python在C级别以缓冲区协议的形式提供了这样的工具。该协议`buffer_protocl`主要有如下接口：

   ```c++
   struct buffer_info {
     void *ptr;
     ssize_t itemsize;
     std::string format;
     ssize_t ndim;
     std::vector<ssize_t> shape;
     std::vector<ssize_t> strides;
   };
   ```

   支持`buffer_protocol`，一个新类型需要在`py::class_`构造函数指定一个特殊的标记`py::buffer_protocol`并且调用方法`def_buffer()`使用lambda函数创建一个`py::buffer_info`描述。

2. 例如有以下新类型`Matrix`，

   ```c++
   class Matrix{
   public:
   	Matrix(size_t rows, size_t cols):m_rows(rows),m_cols(cols){
   		m_data=new float[rows*cols];
   	}
   	float* data() {return m_data;}
   	size_t rows() const {return m_rows;}
   	size_t cols() oonst {return m_cols;}
   private:
   	size_t m_rows,m_cols;
   	float* m_data;
   }
   ```

   下面的绑定代码导出`Matrix`的内容作为一个缓冲对象，可以将Matrix转换为Numpy数组。甚至可以避免复制操作类似`np.array(matrix_instance, copy=false)`。

   ```c++
   py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
   	.def_buffer([](Matrix &m)->py::buffer_info{
   		return py::buffer_info{
   		m.data(),   //pointer to buffer
   		sizeof(float), //size of one scalar
   		py::format_descriptor<float>::format(), //pythonstruct-style format descriptor
   		2, //number of dimensions
   		{m.rows, m.cols()}, //buffer dimensions
   		{sizeof(float)*m.cols(), //strides (in bytes) for each index
   		 sizeof(float)}
   		};
   	});
   ```

   

3. 要创建一个可以将 Python 缓冲区对象作为参数的 C++ 函数，只需使用类型 py::buffer 作为其参数之一。缓冲区可以存在于多种配置中，因此在函数体中通常需要进行一些安全检查。下面，可以看到一个关于如何为 Eigen 双精度矩阵 (Eigen::MatrixXd) 类型定义自定义构造函数的基本示例，该类型支持从兼容的缓冲区对象（例如 NumPy 矩阵）进行初始化。

   ```c++
   /* Bind MatrixXd (or some other Eigen type) to Python */
   typedef Eigen::MatrixXd Matrix;
   
   typedef Matrix::Scalar Scalar;
   constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;
   
   py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
       .def(py::init([](py::buffer b) {
           typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;
   
           /* Request a buffer descriptor from Python */
           py::buffer_info info = b.request();
   
           /* Some sanity checks ... */
           if (info.format != py::format_descriptor<Scalar>::format())
               throw std::runtime_error("Incompatible format: expected a double array!");
   
           if (info.ndim != 2)
               throw std::runtime_error("Incompatible buffer dimension!");
   
           auto strides = Strides(
               info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(Scalar),
               info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(Scalar));
   
           auto map = Eigen::Map<Matrix, 0, Strides>(
               static_cast<Scalar *>(info.ptr), info.shape[0], info.shape[1], strides);
   
           return Matrix(map);
       }));
   ```

   作为参考，此 Eigen 数据类型的 def_buffer() 调用应如下所示：

   ```c++
   .def_buffer([](Matrix &m) -> py::buffer_info {
       return py::buffer_info(
           m.data(),                                /* Pointer to buffer */
           sizeof(Scalar),                          /* Size of one scalar */
           py::format_descriptor<Scalar>::format(), /* Python struct-style format descriptor */
           2,                                       /* Number of dimensions */
           { m.rows(), m.cols() },                  /* Buffer dimensions */
           { sizeof(Scalar) * (rowMajor ? m.cols() : 1),
             sizeof(Scalar) * (rowMajor ? 1 : m.rows()) }
                                                    /* Strides (in bytes) for each index */
       );
    })
   ```

4. 直接访问

   ```c++
   #include <pybind11/pybind11.h>
   #include <pybind11/numpy.h>
   
   namespace py = pybind11;
   
   py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
       py::buffer_info buf1 = input1.request(), buf2 = input2.request();
   
       if (buf1.ndim != 1 || buf2.ndim != 1)
           throw std::runtime_error("Number of dimensions must be one");
   
       if (buf1.size != buf2.size)
           throw std::runtime_error("Input shapes must match");
   
       /* No pointer is passed, so NumPy will allocate the buffer */
       auto result = py::array_t<double>(buf1.size);
   
       py::buffer_info buf3 = result.request();
   
       double *ptr1 = static_cast<double *>(buf1.ptr);
       double *ptr2 = static_cast<double *>(buf2.ptr);
       double *ptr3 = static_cast<double *>(buf3.ptr);
   
       for (size_t idx = 0; idx < buf1.shape[0]; idx++)
           ptr3[idx] = ptr1[idx] + ptr2[idx];
   
       return result;
   }
   
   PYBIND11_MODULE(test, m) {
       m.def("add_arrays", &add_arrays, "Add two NumPy arrays");
   }
   ```

   
