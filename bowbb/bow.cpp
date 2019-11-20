#include<iostream>
#include<map>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<string>
#include<opencv2/ml/ml.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<fstream>
//boost 库
#include<boost/filesystem.hpp>


using namespace cv;
using namespace std;
//定义一个boost库的命名空间
namespace fs = boost::filesystem;
using namespace fs;

void train(int _clusters, string dF, string tF, string tempF, string testF, string rF) {

	string dataFolder = dF;
	string trainFolder = tF;
	string templateFolder = tempF;
	string testFolder = testF;
	string resultFolder = rF;

	//存放所有训练图片的BOW
	map<string, Mat> allsamples_bow;
	//从类目名称到训练图集的映射，关键字可以重复出现
	multimap<string, Mat> train_set;
	// 训练得到的SVM
	CvSVM *stor_svms;
	//类目名称，也就是TRAIN_FOLDER设置的目录名
	vector<string> category_name;
	//类目数目
	int categories_size;
	//用SURF特征构造视觉词库的聚类数目
	int clusters;
	//存放训练图片词典
	Mat vocab;

	Ptr<FeatureDetector> featureDecter = new SurfFeatureDetector();
	Ptr<DescriptorExtractor> descriptorExtractor = new SurfDescriptorExtractor();

	Ptr<BOWKMeansTrainer> bowtrainer = new BOWKMeansTrainer(_clusters);
	Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor;
	Ptr<FlannBasedMatcher> descriptorMacher = new FlannBasedMatcher();

	bowDescriptorExtractor = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMacher);

	//读取训练集
	cout << "读取训练集..." << endl;
	string categor;
	//递归迭代rescursive 直接定义两个迭代器：i为迭代起点（有参数），end_iter迭代终点
	int index = 0;
	for (recursive_directory_iterator i(trainFolder), end_iter; i != end_iter; i++) {
		index++;
		// level == 0即为目录，因为TRAIN__FOLDER中设置如此
		if (i.level() == 0) {
			// 将类目名称设置为目录的名称
			categor = (i->path()).filename().string();
			category_name.push_back(categor);
		}
		else {
			// 读取文件夹下的文件。level 1表示这是一副训练图，通过multimap容器来建立由类目名称到训练图的一对多的映射
			string filename = string(trainFolder) + categor + string("/") + (i->path()).filename().string();
			Mat temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
			pair<string, Mat> p(categor, temp);
			//得到训练集
			train_set.insert(p);
			//cout << p.first << endl;
		}
	}
	categories_size = category_name.size();
	cout << "发现 " << categories_size << "种类别物体..." << endl;

	// 训练图片feature聚类，得出词典
	FileStorage vacab_fs(dataFolder + "svm/" + "vocab.xml", FileStorage::READ);
	//如果之前已经生成好，就不需要重新聚类生成词典
	if (vacab_fs.isOpened()) {
		cout << "图片已经聚类，词典已经存在.." << endl;
		vacab_fs.release();
	}
	else {
		Mat vocab_descriptors;
		vector<KeyPoint>kp;
		// 对于每一幅模板，提取SURF算子，存入到vocab_descriptors中
		multimap<string, Mat> ::iterator i = train_set.begin();
		for (; i != train_set.end(); i++) {
			Mat templ = (*i).second;
			Mat descrip;
			featureDecter->detect(templ, kp);
			descriptorExtractor->compute(templ, kp, descrip);

			//push_back(Mat);在原来的Mat的最后一行后再加几行,元素为Mat时， 其类型和列的数目 必须和矩阵容器是相同的
			vocab_descriptors.push_back(descrip);
		}
		cout << "训练图片开始聚类..." << endl;
		//将每一副图的Surf特征利用add函数加入到bowTraining中去,就可以进行聚类训练了
		bowtrainer->add(vocab_descriptors);
		// 对SURF描述子进行聚类
		vocab = bowtrainer->cluster();
		cout << "聚类完毕，得出词典..." << endl;
		//以文件格式保存词典
		FileStorage file_stor(dataFolder + "svm/" + "vocab.xml", FileStorage::WRITE);
		file_stor << "vocabulary" << vocab;
		file_stor.release();
	}
	//构造bag of words
	cout << "构造bag of words..." << endl;
	FileStorage va_fs(dataFolder + "svm/" + "vocab.xml", FileStorage::READ);
	//如果词典存在则直接读取
	if (va_fs.isOpened()) {
		Mat temp_vacab;
		va_fs["vocabulary"] >> temp_vacab;
		bowDescriptorExtractor->setVocabulary(temp_vacab);
		va_fs.release();
	}
	else {
		//对每张图片的特征点，统计这张图片各个类别出现的频率，作为这张图片的bag of words
		bowDescriptorExtractor->setVocabulary(vocab);
	}

	//如果bow.txt已经存在说明之前已经训练过了，下面就不用重新构造BOW
	string bow_path = dataFolder + "svm/" + string("bow.txt");
	std::ifstream read_file(bow_path);

	//如BOW已经存在，则不需要构造
	if (read_file.is_open()) {
		cout << "BOW 已经准备好..." << endl;
	}
	else {
		// 对于每一幅模板，提取SURF算子，存入到vocab_descriptors中
		multimap<string, Mat> ::iterator i = train_set.begin();

		for (; i != train_set.end(); i++) {
			vector<KeyPoint>kp;
			string cate_nam = (*i).first;
			Mat tem_image = (*i).second;
			Mat imageDescriptor;
			featureDecter->detect(tem_image, kp);

			bowDescriptorExtractor->compute(tem_image, kp, imageDescriptor);
			//push_back(Mat);在原来的Mat的最后一行后再加几行,元素为Mat时， 其类型和列的数目 必须和矩阵容器是相同的
			allsamples_bow[cate_nam].push_back(imageDescriptor);
		}
		//简单输出一个文本，为后面判断做准备
		std::ofstream ous(bow_path);
		ous << "flag";
		cout << "bag of words构造完毕..." << endl;
	}
	//训练分类器
	int flag = 0;
	for (int k = 0; k < categories_size; k++) {
		string svm_file_path = dataFolder + "svm/type/" + category_name[k] + string("SVM.xml");
		FileStorage svm_fil(svm_file_path, FileStorage::READ);
		//判断训练结果是否存在
		if (svm_fil.isOpened())
		{
			svm_fil.release();
			continue;
		}
		else
		{
			flag = -1;
			break;
		}
	}

	//如果训练结果已经存在则不需要重新训练
	if (flag != -1) {
		cout << "分类器已经训练完毕..." << endl;
	}
	else {
		stor_svms = new CvSVM[categories_size];
		//设置训练参数
		SVMParams svmParams;
		svmParams.svm_type = CvSVM::C_SVC;
		svmParams.kernel_type = CvSVM::LINEAR;
		svmParams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

		cout << "训练分类器..." << endl;
		for (int i = 0; i < categories_size; i++) {
			Mat tem_Samples(0, allsamples_bow.at(category_name[i]).cols, allsamples_bow.at(category_name[i]).type());
			Mat responses(0, 1, CV_32SC1);
			tem_Samples.push_back(allsamples_bow.at(category_name[i]));
			Mat posResponses(allsamples_bow.at(category_name[i]).rows, 1, CV_32SC1, Scalar::all(1));
			responses.push_back(posResponses);

			for (auto itr = allsamples_bow.begin(); itr != allsamples_bow.end(); ++itr) {
				if (itr->first == category_name[i]) {
					continue;
				}
				tem_Samples.push_back(itr->second);
				Mat response(itr->second.rows, 1, CV_32SC1, Scalar::all(-1));
				responses.push_back(response);
			}

			stor_svms[i].train(tem_Samples, responses, Mat(), Mat(), svmParams);
			//存储svm
			string svm_filename = dataFolder + "svm/type/" + category_name[i] + string("SVM.xml");
			stor_svms[i].save(svm_filename.c_str());
		}
		cout << "分类器训练完毕..." << endl;
	}
}

String categoryImage(string trainPicPath,string dataFolder) {

	vector<string> category_name;
	int categoryNameSize = 0;

	for (recursive_directory_iterator i(dataFolder + "svm/type/"), end_iter; i != end_iter; i++) {
		string categor = (i->path()).filename().string();

		int last_index = categor.find_last_of(".");
		string name = categor.substr(0, last_index - 3);

		category_name.push_back(name);
	}
	categoryNameSize = category_name.size();

	Mat gray_pic;
	string prediction_category;
	float curConfidence;

	//读取图片
	cout << trainPicPath << endl;
	Mat input_pic = imread(trainPicPath);
	cvtColor(input_pic, gray_pic, CV_BGR2GRAY);

	// 提取BOW描述子
	vector<KeyPoint>kp;
	Mat test;

	Ptr<FeatureDetector> featureDecter = new SurfFeatureDetector();

	Ptr<DescriptorExtractor> descriptorExtractor = new SurfDescriptorExtractor();

	Ptr<FlannBasedMatcher> descriptorMacher = new FlannBasedMatcher();
	Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMacher);

	FileStorage va_fs(dataFolder + "svm/" + "vocab.xml", FileStorage::READ);
	//如果词典存在则直接读取
	if (va_fs.isOpened()) {
		Mat temp_vacab;
		va_fs["vocabulary"] >> temp_vacab;
		bowDescriptorExtractor->setVocabulary(temp_vacab);
		va_fs.release();
	}

	featureDecter->detect(gray_pic, kp);
	bowDescriptorExtractor->compute(gray_pic, kp, test);

	int sign = 0;
	float best_score = -2.0f;

	for (int i = 0; i < categoryNameSize; i++) {
		string cate_na = category_name[i];

		string f_path = dataFolder + "svm/type/" + cate_na + string("SVM.xml");
		FileStorage svm_fs(f_path, FileStorage::READ);
		//读取SVM.xml+99
		if (svm_fs.isOpened()) {
			svm_fs.release();
			CvSVM st_svm;
			st_svm.load(f_path.c_str());

			if (sign == 0) {
				float score_Value = st_svm.predict(test, true);
				float class_Value = st_svm.predict(test, false);
				sign = (score_Value < 0.0f) == (class_Value < 0.0f) ? 1 : -1;
			}
			curConfidence = sign * st_svm.predict(test, true);
		}
		if (curConfidence > best_score) {
			best_score = curConfidence;
			prediction_category = cate_na;
		}
	}
	return prediction_category;
}

void categoryBySvm(string dataFolder, string testFolder,bool flag) {
	cout << "物体分类开始..." << endl;

	directory_iterator begin_train(testFolder);
	directory_iterator end_train;

	string resultFolder = dataFolder + "/result_image/";
	string templateFolder = dataFolder + "/templates/";

	map<string, Mat> result_objects;
	directory_iterator begin_iter(templateFolder);
	directory_iterator end_iter;
	//获取该目录下的所有文件名
	for (; begin_iter != end_iter; ++begin_iter) {
		string imageName = begin_iter->path().filename().string();

		string filename = templateFolder + imageName;

		int last_index = imageName.find_last_of(".");
		string name = imageName.substr(0, last_index);

		//读入模板图片
		Mat image = imread(filename);
		//Mat templ_image;
		//存储原图模板
		result_objects[name] = image;
	}


	for (; begin_train != end_train; ++begin_train) {
		//获取该目录下的图片名
		string trainPicName = (begin_train->path()).filename().string();
		string trainPicPath = testFolder + string("/") + (begin_train->path()).filename().string();

		string prediction_category = categoryImage(trainPicPath, dataFolder);

		Mat input_pic = imread(trainPicPath);

		//将图片写入相应的文件夹下
		directory_iterator begin_iterater(resultFolder);
		directory_iterator end_iterator;
		//获取该目录下的文件名
		for (; begin_iterater != end_iterator; ++begin_iterater) {
			if (begin_iterater->path().filename().string() == prediction_category) {
				string filename = resultFolder + prediction_category + string("/") + trainPicName;
				imwrite(filename, input_pic);
			}
		}
		cout << "这张图属于： " << prediction_category << endl;
		//显示输出
		if (flag) {
			imshow("输入图片：", input_pic);

			namedWindow("Dectect Object");
			
			imshow("Dectect Object", result_objects[prediction_category]);
			waitKey(0);
		}
	}
}

int main(void) {

	int clusters = 1000;

	string dataFolder = "D:/project data/data/";
	string trainFolder = "D:/project data/data/train_images/";
	string templateFolder = "D:/project data/data/templates/";
	string testFolder = "D:/project data/data/test_image";
	string resultFolder = "D:/project data/data/result_image/";


	//string dataFolder = "data/";
	//string trainFolder = "data/train_images/";
	//string templateFolder = "data/templates/";
	//string testFolder = "data/test_image";
	//string resultFolder = "data/result_image/";

	train(clusters, dataFolder, trainFolder, templateFolder, testFolder, resultFolder);

	//将测试图片分类
	categoryBySvm(dataFolder, testFolder, true);
	return 0;
}