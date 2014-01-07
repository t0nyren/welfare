#include <iostream>
#include "vladIndex.h"
using namespace std;

int main(int argc, char** argv){
	VladIndex index;
	while(true){
		string img;
		cout<<"Input img: ";
		cin>>img;
		int id = index.predict(img.data());
		cout<<"Predicted class: "<<id<<endl;
	}
	return 0;
}