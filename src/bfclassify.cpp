#include <iostream>
#include "vladIndex.h"
#include <dirent.h>
#include<sys/stat.h>
#include<sys/types.h>
using namespace std;

int main(int argc, char** argv){
	
	/*while(true){
		string img;
		cout<<"Input img: ";
		cin>>img;
		string classname = index.predict(img.data());
		cout<<"Predicted class: "<<classname<<endl;
	}*/
	bool isDetect = 1;
	if (argc < 2){
		cout<<"usage: search [directory] isDetect"<<endl;
		return -1;
	}
	if (argc ==3)
		isDetect = atoi(argv[2]);
	DIR *pDIR;
	struct dirent *entry;
	struct stat buf;
	pDIR = opendir(argv[1]);
	if(( pDIR=opendir(argv[1])) == NULL){
		std::cout<<"cannot open input dir"<<std::endl;
		exit(1);
	}
	entry = readdir(pDIR);
	int count = 0;
	int correct = 0;
	VladIndex index;
	cout<<"Precision: "<<index.evalPrecision()<<endl;
	while(entry != NULL)
	{
		if(0 != strcmp( ".", entry->d_name) && //Skip these dir
		   0 != strcmp( "..", entry->d_name) )
		{
			char * name = entry->d_name;
			stat(name, &buf);
			std::cout << name<<std::endl;
			std::string s2 = name;
			cout<<name<<endl;
			DIR* pDIR2;
			std::string s3 = argv[1];
			std::string origin_path = s3 + '/' + s2;
			pDIR2 = opendir(origin_path.data());
			struct dirent *entry2;
			entry2 = readdir(pDIR2);
			while(entry2 != NULL){
				if(0 != strcmp( ".", entry2->d_name) && 0 != strcmp( "..", entry2->d_name) )
				{
					std::string img_origin_path = origin_path + '/' + entry2->d_name;
					string classname = index.predict(img_origin_path.data(), isDetect);
					if(classname != ""){
						cout<<"predicted class: "<<classname<<" truth: "<<entry->d_name<<endl;
						if(strcmp(classname.data(), entry->d_name)==0){
							correct++;
						}
						count++;
					}
				}
				entry2 = readdir(pDIR2);
			}
			closedir(pDIR2);
		}
		entry = readdir(pDIR);             //Next file in directory        
	}
	cout<<"precision: "<<(double)correct/count<<endl;
	return 0;
}