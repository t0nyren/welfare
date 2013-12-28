all: bin/classify bin/align_single

CV_INCLUDE = -I"/home/zhouping/tonyren/cv/include"
VL_INCLUDE = -I"/home/zhouping/tonyren/vlfeat"
VL_LINK = -L"./lib" -lvl
CV_LINK = -L"/home/zhouping/tonyren/cv/lib"  -lopencv_core -lopencv_highgui  -lopencv_objdetect -lopencv_contrib -lopencv_legacy

bin/classify: build/main.o build/classifier.o build/detector.o build/mblbp-detect.o build/flandmark_detector.o build/liblbp.o
	g++ -o bin/classify build/main.o build/classifier.o build/detector.o build/mblbp-detect.o build/flandmark_detector.o build/liblbp.o $(VL_LINK) $(CV_LINK)

bin/align_single: build/align_single.o build/flandmark_detector.o build/liblbp.o
	g++ -o ./bin/align_single ./build/align_single.o ./build/flandmark_detector.o ./build/liblbp.o $(CV_LINK)

build/main.o: src/main.cpp
	g++ -c ./src/main.cpp -o ./build/main.o $(VL_INCLUDE)

build/classifier.o: src/classifier.cpp
	g++ -c ./src/classifier.cpp -o ./build/classifier.o $(VL_INCLUDE)

build/align_single.o: src/align_single.cpp 
	g++ -c src/align_single.cpp -o build/align_single.o 

build/detector.o: src/detector.cpp
	g++ -c src/detector.cpp -o build/detector.o

build/flandmark_detector.o: src/flandmark_detector.cpp
	g++ -c src/flandmark_detector.cpp -o build/flandmark_detector.o

build/liblbp.o: ./src/liblbp.cpp
	g++ -c src/liblbp.cpp -o build/liblbp.o
	
build/mblbp-detect.o: src/mblbp-detect.h src/mblbp-detect.cpp
	g++ -c src/mblbp-detect.cpp -o build/mblbp-detect.o
clean:
	rm ./build/*.o
	rm ./bin/classify
	rm ./bin/align_single
