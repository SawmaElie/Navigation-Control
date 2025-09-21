#pragma once
#ifndef CIRCLE__CIRCLE_H_
#define CIRCLE__CIRCLE_H_
#include <iostream >
#include "rec/robotino/api2/all.h"
using namespace rec::robotino::api2;
#define PI 3.1415f

class MyCom : public Com {
public:
	MyCom() : Com() {}
	void errorEvent(const char* errorString) {
		std::cerr << "Error: " << errorString << std::endl;
	}
	void connectedEvent() {
		std::cout << "Connected." << std::endl;
	}
	void connectionClosedEvent() {
		std::cout << "Connection closed." << std::endl;
	}
};
#endif // CIRCLE__CIRCLE_H_
