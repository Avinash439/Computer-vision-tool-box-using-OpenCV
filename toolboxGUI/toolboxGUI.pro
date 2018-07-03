#-------------------------------------------------
#
# Project created by QtCreator 2017-04-25T15:40:23
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = toolboxGUI
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += main.cpp\
        mainwindow.cpp \
    functions.cpp \
    histogram.cpp \
    laplacian.cpp \
    cameracalibrator.cpp

HEADERS  += mainwindow.h \
    histogram.h \
    laplacian.h \
    cameracalibrator.h

FORMS    += mainwindow.ui
QT += widgets
DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x000000

INCLUDEPATH += C:\opencv\opencv\mybuild\install\include
LIBS += -LC:\opencv\opencv\mybuild\install\x86\mingw\lib \
    -lopencv_core320.dll \
    -lopencv_highgui320.dll \
    -lopencv_imgcodecs320.dll \
    -lopencv_imgproc320.dll \
    -lopencv_features2d320.dll \
    -lopencv_xfeatures2d320.dll \
    -lopencv_calib3d320.dll
