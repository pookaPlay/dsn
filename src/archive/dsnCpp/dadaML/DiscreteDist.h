#if !defined(DiscreteDist_H__)
#define DiscreteDist_H__

#include <random>

class DiscreteDist {
private:
	std::default_random_engine generator;
	std::discrete_distribution<> distribution;
	static std::discrete_distribution<> makeDistribution(vector<double> &values)
	{
		std::array<double, 7> distArray;
		distArray.fill(1);
		//distArray[0] = 0;
		//distArray[loadSide] = loadAmount;
		return{ std::begin(distArray), std::end(distArray) };
	}
public:
	DiscreteDist(vector<double> &values) :
		generator{},
		distribution{ makeDistribution(values) }
	{}
	int roll() {
		return distribution(generator);
	}
};

#endif
