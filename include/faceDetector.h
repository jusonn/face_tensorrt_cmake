#pragma once
// MathFuncsLib.h

#include "checkBlacklist.h"
#include "ocrCommon.h"

namespace OcrDetectFunc
{
	class DetectFunc
	{
	public:
		// Returns a + b
		static double Add(double a, double b);

		// Returns a - b
		static double Subtract(double a, double b);

		// Returns a * b
		static double Multiply(double a, double b);

		// Returns a / b
		static double Divide(double a, double b);
	};
}