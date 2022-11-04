# Qt Basic




## Qt连接MySQL

1. **核心代码**

```c++
#include <QSqlDatabase>
#include <QDebug>
#include <QMessageBox>
#include <QSqlError>

    //连接数据库
    QSqlDatabase db=QSqlDatabase::addDatabase("QMYSQL");
    db.setHostName("127.0.0.1");

    db.setDatabaseName("docmanager");
    db.setUserName("root");
    db.setPassword("123156");

    if(!db.open())
    {
        QString errInfo=db.lastError().text();
        qDebug()<<errInfo;
    }else{
        qDebug()<<"info: create connection to db is successful!";
    }

    db.close();
```

2. **"Authentication plugin 'caching_sha2_password"错误**

如果是MySQL8.0版本，需要修改登录数据库的授权模式。首先打开MySQL的命令行，输入数据库密码。

```shell
ALTER USER 'root'@'localhost' IDENTIFIED BY 'password' PASSWORD EXPIRE NEVER;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';
FLUSH PRIVILEGES;
ALTER USER 'root'@'localhost' IDENTIFIED BY '新密码'
```

3. **"QSqlDatabase: QMYSQL driver not loaded"错误**

报该错误是因为缺少MySQL的驱动`libmysql.dll`和`libmysql.lib`。

- **下载驱动包**

下载地址：https://dev.mysql.com/downloads/。选择`C API(libmysqlclient)`。下载64位版本(`mysql-connector-c-6.1.11-winx64.zip`)或32位版本(`mysql-connector-c-6.1.11-win32`.zip)，并解压。

- **复制lib文件夹下的`libmysql.dll`和`libmydql.lib`**

根据build环境选择将`libmysql.dll`和`libmydql.lib`复制到`X:\Qt\5.XX\mingwXX_64\bin`目录下。

4. **QSqlDatabase: available drivers: QSQLITE QMARIADB QMYSQL QMYSQL3 QODBC QODBC3 QPSQL QPSQL7**错误

报错原因是：缺少驱动`qsqlmysql.dll`。

(1). 打开qt源码`X:\Qt\5.XX\Src\qtbase\src\plugins\sqldrivers\mysql\mysql.pro`。

(2). 配置`mysql.pro`文件

    - 注释`QMAKE_USE += mysql`

    - 设置生成文件所在路径 `DESTDIR= …/mysqldll`
    
    - 设置库文件目录为下载驱动的目录文件。 `LIBS += "X:\XX\mysql-connector-c-6.1.11-winx64\lib\libmysql.lib`和`INCLUDEPATH += X:\XX\mysql-connector-c-6.1.11-winx64\include`

(3). 配置`qsqldriverbase.pri`文件

    - 注释`include($$shadowed.....)`

    - 替换为`include(.configure.pri)`

(4). 选择`release`版本，编译生成`qsqlmysql.dll`文件

(5). 将生成的`qsqlmysql.dll`文件复制到`X:\Qt\5.XX.X\mingwXX_64\plugins\sqldrivers`目录，重启QCreator即可使用核心代码测试数据库链接是否成功。